import re
from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio

from src.audio.io import AudioSource, load_audio_source


WHITESPACE_PATTERN = re.compile(r"\s+")

# Registry of supported ASR backbones. Adding a new backbone means adding one
# entry here (plus a Runner class if it needs a new inference family) --
# nothing else in the pipeline (manifests, VAD policy, decoding scripts)
# needs to change.
#
# "wav2vec2-base-960h" is the backbone used in the baseline experiment; its
# behavior is unchanged by this registry.
BACKBONE_REGISTRY = {
    "wav2vec2-base-960h": {"family": "wav2vec2_torchaudio", "hf_id": None},
    "whisper-base": {"family": "whisper_hf", "hf_id": "openai/whisper-base"},
    "whisper-small": {"family": "whisper_hf", "hf_id": "openai/whisper-small"},
    "wav2vec2-xlsr-korean": {"family": "wav2vec2_hf", "hf_id": "kresnik/wav2vec2-large-xlsr-korean"},
    # Follow-up study backbones. All three emit LibriSpeech orthography
    # (lowercase, no punctuation, no digits), so CER needs no normalizer and
    # the cross-architecture comparison is not confounded by formatting --
    # which is exactly why Whisper is excluded from that axis.
    "hubert-large-ls960-ft": {"family": "hf_ctc", "hf_id": "facebook/hubert-large-ls960-ft"},
    "emformer-rnnt-base": {"family": "rnnt_torchaudio", "hf_id": None},
    "speechbrain-transformer": {
        "family": "speechbrain_enc_dec",
        "hf_id": "speechbrain/asr-transformer-transformerlm-librispeech",
    },
}


@dataclass
class AsrSettings:
    model_name: str
    device: str
    device_index: int
    hf_id: Optional[str] = None
    language: Optional[str] = None
    physical_device_index: Optional[int] = None


def apply_asr_overrides(config, model_name=None, language=None):
    """Point one config at a different backbone for a single run.

    The cross-architecture axis needs the same dataset decoded by several
    backbones. Hand-editing conf/base.yaml between runs is what previously let
    two backbones write over each other's eval CSVs, so every entry point takes
    --model / --language instead and the run tag follows automatically.

    language accepts "none"/"null"/"auto" to clear it, which is what the CTC
    backbones want (they ignore it) and what Whisper needs for auto-detection.
    """
    if model_name is not None:
        if model_name not in BACKBONE_REGISTRY:
            raise ValueError(
                "Unsupported ASR model: %s. Supported: %s" % (model_name, ", ".join(sorted(BACKBONE_REGISTRY)))
            )
        config["asr"]["model_name"] = model_name

    if language is not None:
        config["asr"]["language"] = None if language.strip().lower() in ("none", "null", "auto") else language

    return config


def get_asr_model_tag(config):
    model_name = config["asr"]["model_name"]
    return model_name.replace("-", "_")


def get_device_indices(config):
    device_index = config["asr"]["device_index"]
    if isinstance(device_index, list):
        return [int(value) for value in device_index]
    return [int(device_index)]


def resolve_device(device, device_index):
    if device == "cuda":
        return "cuda:%d" % device_index
    return "cpu"


def decode_tokens(token_ids, labels):
    blank_id = 0
    pieces = []
    previous_id = None

    for token_id in token_ids:
        if token_id == previous_id:
            continue
        previous_id = token_id
        if token_id == blank_id:
            continue
        token = labels[token_id]
        if token == "|":
            pieces.append(" ")
        else:
            pieces.append(token)

    text = "".join(pieces)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def _load_and_resample(source, target_sample_rate):
    """Load a whole file or a window inside one (see io.AudioSource) and resample."""
    audio, sample_rate = load_audio_source(source)
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return waveform, sample_rate


class Wav2Vec2Runner:
    """CTC backbone used in the baseline experiment. Unchanged from the baseline."""

    def __init__(self, settings):
        self.settings = settings
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.labels = self.bundle.get_labels()
        self.sample_rate = self.bundle.sample_rate
        self.device = resolve_device(settings.device, settings.device_index)

        self.model = self.bundle.get_model().to(self.device)
        self.model.eval()

    def transcribe(self, source):
        waveform, sample_rate = _load_and_resample(source, self.sample_rate)
        waveform = waveform.to(self.device)

        with torch.inference_mode():
            emissions, _ = self.model(waveform)

        token_ids = torch.argmax(emissions, dim=-1)[0].detach().cpu().tolist()
        hypothesis = decode_tokens(token_ids, self.labels)

        return {
            "hypothesis": hypothesis,
            "language": "en",
            "language_probability": 1.0,
            "duration_sec": waveform.shape[-1] / sample_rate,
            "segment_count": 1,
        }


class WhisperRunner:
    """Attention-based encoder-decoder backbone, for the cross-architecture
    generalization analysis (extension A)."""

    SAMPLE_RATE = 16000
    # LibriSpeech/KsponSpeech utterances are short sentences; this bounds the
    # worst case where generate() would otherwise ramble toward its internal
    # max length (448 tokens) on a bad/non-speech input instead of hitting EOS.
    MAX_NEW_TOKENS = 256

    def __init__(self, settings):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.settings = settings
        self.device = resolve_device(settings.device, settings.device_index)

        self.processor = WhisperProcessor.from_pretrained(settings.hf_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(settings.hf_id)
        self.model.to(self.device)
        self.model.eval()

    def _generate(self, input_features):
        generate_kwargs = {"task": "transcribe", "max_new_tokens": self.MAX_NEW_TOKENS}
        if self.settings.language:
            generate_kwargs["language"] = self.settings.language
        with torch.inference_mode():
            return self.model.generate(input_features, **generate_kwargs)

    def transcribe(self, source):
        waveform, sample_rate = _load_and_resample(source, self.SAMPLE_RATE)
        audio = waveform.squeeze(0).numpy()

        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        predicted_ids = self._generate(input_features)
        hypothesis = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        return {
            "hypothesis": hypothesis,
            "language": self.settings.language or "auto",
            "language_probability": 1.0,
            "duration_sec": waveform.shape[-1] / sample_rate,
            "segment_count": 1,
        }

    def transcribe_batch(self, sources):
        waveforms = []
        durations = []
        for source in sources:
            waveform, sample_rate = _load_and_resample(source, self.SAMPLE_RATE)
            waveforms.append(waveform.squeeze(0).numpy())
            durations.append(waveform.shape[-1] / sample_rate)

        # WhisperFeatureExtractor pads/trims every clip to a fixed 30s log-mel
        # window, so a batch of differently-sized clips produces a uniformly
        # shaped tensor with no extra padding/attention-mask bookkeeping needed.
        inputs = self.processor(waveforms, sampling_rate=self.SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        predicted_ids = self._generate(input_features)
        hypotheses = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return [
            {
                "hypothesis": hypothesis.strip(),
                "language": self.settings.language or "auto",
                "language_probability": 1.0,
                "duration_sec": duration,
                "segment_count": 1,
            }
            for hypothesis, duration in zip(hypotheses, durations)
        ]


class HFWav2Vec2Runner:
    """Generic HuggingFace CTC wav2vec2 backbone, for non-English backbones
    (e.g. Korean xlsr) used in the cross-lingual generalization analysis
    (extension C)."""

    SAMPLE_RATE = 16000

    def __init__(self, settings):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.settings = settings
        self.device = resolve_device(settings.device, settings.device_index)

        self.processor = Wav2Vec2Processor.from_pretrained(settings.hf_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(settings.hf_id)
        self.model.to(self.device)
        self.model.eval()

    def transcribe(self, source):
        waveform, sample_rate = _load_and_resample(source, self.SAMPLE_RATE)
        audio = waveform.squeeze(0).numpy()

        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)

        with torch.inference_mode():
            logits = self.model(input_values).logits

        token_ids = torch.argmax(logits, dim=-1)[0].detach().cpu().tolist()
        hypothesis = self.processor.decode(token_ids).strip()

        return {
            "hypothesis": hypothesis,
            "language": self.settings.language or "ko",
            "language_probability": 1.0,
            "duration_sec": waveform.shape[-1] / sample_rate,
            "segment_count": 1,
        }

    def transcribe_batch(self, sources):
        waveforms = []
        durations = []
        for source in sources:
            waveform, sample_rate = _load_and_resample(source, self.SAMPLE_RATE)
            waveforms.append(waveform.squeeze(0).numpy())
            durations.append(waveform.shape[-1] / sample_rate)

        # Variable-length clips need padding; attention_mask tells the model
        # (and, implicitly, the CTC argmax below) which frames are real audio.
        inputs = self.processor(
            waveforms, sampling_rate=self.SAMPLE_RATE, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            logits = self.model(input_values, attention_mask=attention_mask).logits

        if attention_mask is not None:
            output_lengths = self.model._get_feat_extract_output_lengths(attention_mask.sum(-1))
        else:
            output_lengths = None

        results = []
        for i, duration in enumerate(durations):
            length = int(output_lengths[i]) if output_lengths is not None else logits.shape[1]
            token_ids = torch.argmax(logits[i, :length], dim=-1).detach().cpu().tolist()
            hypothesis = self.processor.decode(token_ids).strip()
            results.append(
                {
                    "hypothesis": hypothesis,
                    "language": self.settings.language or "ko",
                    "language_probability": 1.0,
                    "duration_sec": duration,
                    "segment_count": 1,
                }
            )
        return results



class HFCtcRunner:
    """Generic HuggingFace CTC backbone (HuBERT, wav2vec2) for English.

    Separate from HFWav2Vec2Runner, which hardcodes Korean as its reported
    language for the cross-lingual work this study does not use.
    """

    SAMPLE_RATE = 16000

    def __init__(self, settings):
        from transformers import AutoModelForCTC, AutoProcessor

        self.settings = settings
        self.device = resolve_device(settings.device, settings.device_index)

        self.processor = AutoProcessor.from_pretrained(settings.hf_id)
        self.model = AutoModelForCTC.from_pretrained(settings.hf_id)
        self.model.to(self.device)
        self.model.eval()

    def _decode_logits(self, logits, length=None):
        sliced = logits if length is None else logits[:length]
        token_ids = torch.argmax(sliced, dim=-1).detach().cpu().tolist()
        return self.processor.decode(token_ids).strip()

    def _result(self, hypothesis, duration_sec):
        return {
            "hypothesis": hypothesis,
            "language": self.settings.language or "en",
            "language_probability": 1.0,
            "duration_sec": duration_sec,
            "segment_count": 1,
        }

    def transcribe(self, source):
        waveform, sample_rate = _load_and_resample(source, self.SAMPLE_RATE)
        inputs = self.processor(
            waveform.squeeze(0).numpy(), sampling_rate=sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)
        with torch.inference_mode():
            logits = self.model(input_values).logits
        return self._result(self._decode_logits(logits[0]), waveform.shape[-1] / sample_rate)

    def transcribe_batch(self, sources):
        waveforms = []
        durations = []
        for source in sources:
            waveform, sample_rate = _load_and_resample(source, self.SAMPLE_RATE)
            waveforms.append(waveform.squeeze(0).numpy())
            durations.append(waveform.shape[-1] / sample_rate)

        inputs = self.processor(
            waveforms, sampling_rate=self.SAMPLE_RATE, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            logits = self.model(input_values, attention_mask=attention_mask).logits

        lengths = (
            self.model._get_feat_extract_output_lengths(attention_mask.sum(-1))
            if attention_mask is not None
            else None
        )
        return [
            self._result(
                self._decode_logits(logits[i], int(lengths[i]) if lengths is not None else None),
                duration,
            )
            for i, duration in enumerate(durations)
        ]


class EmformerRnntRunner:
    """Streaming Transducer (torchaudio Emformer RNN-T, LibriSpeech).

    The study's Transducer arm, and the only backbone here whose latency can be
    measured rather than computed: it is a streaming model, so E5 can report a
    real emission time against the analytic min_silence + trailing_pad budget.
    Its 4,096-token SentencePiece vocabulary holds no digits, punctuation or
    uppercase, so it needs no text normalization.
    """

    BEAM_WIDTH = 10

    def __init__(self, settings):
        from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH as bundle

        self.settings = settings
        self.device = resolve_device(settings.device, settings.device_index)
        self.sample_rate = bundle.sample_rate

        self.feature_extractor = bundle.get_feature_extractor()
        self.token_processor = bundle.get_token_processor()
        self.decoder = bundle.get_decoder()
        self.decoder.model.to(self.device)
        self.decoder.model.eval()

    def transcribe(self, source):
        waveform, sample_rate = _load_and_resample(source, self.sample_rate)
        audio = waveform.squeeze(0)
        duration_sec = audio.shape[-1] / sample_rate

        if audio.numel() == 0:
            return {
                "hypothesis": "",
                "language": "en",
                "language_probability": 1.0,
                "duration_sec": duration_sec,
                "segment_count": 1,
            }

        features, lengths = self.feature_extractor(audio)
        with torch.inference_mode():
            hypotheses = self.decoder(
                features.unsqueeze(0).to(self.device),
                lengths.to(self.device),
                beam_width=self.BEAM_WIDTH,
            )

        # RNNTBeamSearch returns the beam best-first; each hypothesis is a
        # tuple whose first element is the token list.
        hypothesis = self.token_processor(hypotheses[0][0]).strip() if hypotheses else ""
        return {
            "hypothesis": hypothesis,
            "language": "en",
            "language_probability": 1.0,
            "duration_sec": duration_sec,
            "segment_count": 1,
        }


class SpeechBrainEncDecRunner:
    """Attention encoder-decoder (SpeechBrain transformer, LibriSpeech).

    The attention arm of the architecture axis. Trained on LibriSpeech text, so
    unlike Whisper it emits the same orthography as the CTC backbones.
    """

    SAMPLE_RATE = 16000

    def __init__(self, settings):
        from speechbrain.inference.ASR import EncoderDecoderASR

        self.settings = settings
        self.device = resolve_device(settings.device, settings.device_index)
        self.model = EncoderDecoderASR.from_hparams(
            source=settings.hf_id,
            savedir=f"artifacts/speechbrain/{settings.hf_id.replace('/', '__')}",
            run_opts={"device": self.device},
        )

    def _result(self, hypothesis, duration_sec):
        return {
            "hypothesis": (hypothesis or "").strip().lower(),
            "language": self.settings.language or "en",
            "language_probability": 1.0,
            "duration_sec": duration_sec,
            "segment_count": 1,
        }

    def transcribe(self, source):
        waveform, sample_rate = _load_and_resample(source, self.SAMPLE_RATE)
        duration_sec = waveform.shape[-1] / sample_rate
        with torch.inference_mode():
            hypotheses, _ = self.model.transcribe_batch(
                waveform.to(self.device), torch.tensor([1.0], device=self.device)
            )
        return self._result(hypotheses[0], duration_sec)

    def transcribe_batch(self, sources):
        waveforms = []
        durations = []
        for source in sources:
            waveform, sample_rate = _load_and_resample(source, self.SAMPLE_RATE)
            waveforms.append(waveform.squeeze(0))
            durations.append(waveform.shape[-1] / sample_rate)

        # SpeechBrain takes a padded batch plus each row's fraction of the
        # padded length, so the encoder can mask the padding away.
        max_length = max(int(w.shape[-1]) for w in waveforms)
        batch = torch.zeros(len(waveforms), max_length)
        relative_lengths = torch.zeros(len(waveforms))
        for index, waveform in enumerate(waveforms):
            batch[index, : waveform.shape[-1]] = waveform
            relative_lengths[index] = waveform.shape[-1] / max_length

        with torch.inference_mode():
            hypotheses, _ = self.model.transcribe_batch(
                batch.to(self.device), relative_lengths.to(self.device)
            )
        return [self._result(h, d) for h, d in zip(hypotheses, durations)]


RUNNER_CLASSES = {
    "wav2vec2_torchaudio": Wav2Vec2Runner,
    "whisper_hf": WhisperRunner,
    "wav2vec2_hf": HFWav2Vec2Runner,
    "hf_ctc": HFCtcRunner,
    "rnnt_torchaudio": EmformerRnntRunner,
    "speechbrain_enc_dec": SpeechBrainEncDecRunner,
}


def create_asr_runners(config):
    model_name = config["asr"]["model_name"]
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(
            "Unsupported ASR model: %s. Supported: %s" % (model_name, ", ".join(sorted(BACKBONE_REGISTRY)))
        )

    backbone = BACKBONE_REGISTRY[model_name]
    runner_cls = RUNNER_CLASSES[backbone["family"]]
    language = config["asr"].get("language")

    if config["asr"]["device"] == "cpu":
        settings = AsrSettings(
            model_name=model_name,
            hf_id=backbone["hf_id"],
            device="cpu",
            device_index=0,
            language=language,
        )
        return [runner_cls(settings)]

    runners = []
    # CUDA_VISIBLE_DEVICES remaps selected physical GPUs to local cuda:0..N-1 ordinals.
    for local_device_index, physical_device_index in enumerate(get_device_indices(config)):
        settings = AsrSettings(
            model_name=model_name,
            hf_id=backbone["hf_id"],
            device=config["asr"]["device"],
            device_index=local_device_index,
            physical_device_index=physical_device_index,
            language=language,
        )
        runners.append(runner_cls(settings))
    return runners
