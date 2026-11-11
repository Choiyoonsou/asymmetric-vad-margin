from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from src.audio.io import concatenate_segments


@dataclass
class SileroVadSettings:
    threshold: float
    min_silence_duration_ms: int
    speech_pad_ms: int
    max_speech_duration_s: float
    target_sample_rate: int


def resample_audio(audio, source_rate, target_rate):
    if source_rate == target_rate:
        return audio.astype(np.float32)

    source_positions = np.arange(len(audio), dtype=np.float32)
    target_length = int(round(len(audio) * (target_rate / source_rate)))
    if target_length <= 1:
        return np.array([], dtype=np.float32)

    target_positions = np.linspace(0, len(audio) - 1, num=target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def segments_from_frame_flags(
    flags,
    frame_length_samples,
    audio_length,
    min_silence_duration_ms,
    speech_pad_ms,
    max_speech_duration_s,
    sample_rate,
):
    """Turn per-frame speech decisions into Silero-shaped sample segments.

    WebRTC and pyannote emit raw frame decisions, Silero emits finished
    segments. Running both through one post-processor -- same silence merging,
    same speech padding, same maximum segment length -- keeps E4 a comparison
    of the detectors rather than of three different smoothing heuristics.
    """
    min_silence_frames = max(1, int(round((min_silence_duration_ms / 1000.0) * sample_rate / frame_length_samples)))
    pad_samples = max(0, int(round((speech_pad_ms / 1000.0) * sample_rate)))
    max_speech_samples = (
        int(round(max_speech_duration_s * sample_rate)) if max_speech_duration_s and max_speech_duration_s > 0 else None
    )

    raw = []
    start_frame = None
    silence_run = 0
    for index, is_speech in enumerate(flags):
        if is_speech:
            if start_frame is None:
                start_frame = index
            silence_run = 0
            continue
        if start_frame is None:
            continue
        silence_run += 1
        if silence_run >= min_silence_frames:
            raw.append((start_frame, index - silence_run + 1))
            start_frame = None
            silence_run = 0
    if start_frame is not None:
        raw.append((start_frame, len(flags)))

    segments = []
    for start_frame, end_frame in raw:
        start = max(0, start_frame * frame_length_samples - pad_samples)
        end = min(audio_length, end_frame * frame_length_samples + pad_samples)
        if end <= start:
            continue
        if max_speech_samples is None:
            segments.append({"start": int(start), "end": int(end)})
            continue
        # Split anything longer than the cap, mirroring Silero's
        # max_speech_duration_s so no backbone sees an outsized chunk.
        cursor = start
        while end - cursor > max_speech_samples:
            segments.append({"start": int(cursor), "end": int(cursor + max_speech_samples)})
            cursor += max_speech_samples
        if end > cursor:
            segments.append({"start": int(cursor), "end": int(end)})

    return segments


class BaseVadRunner:
    """Common surface every VAD in the registry presents.

    name          identifies the detector in output paths and decode records.
    lookahead_ms  audio the detector must see beyond a boundary before it can
                  report it. It enters the algorithmic latency budget
                  (min_silence + trailing_pad + lookahead), which is what E5
                  trades margin accuracy against.
    """

    name = "base"
    lookahead_ms = 0.0

    def detect_segments(self, audio, sample_rate):
        raise NotImplementedError

    def reconstruct_audio(self, audio, segments):
        pieces = [audio[segment["start"] : segment["end"]] for segment in segments]
        return concatenate_segments(pieces)


class SileroVadRunner(BaseVadRunner):
    name = "silero"
    # Silero v5 consumes 512-sample windows at 16 kHz; a boundary is only
    # reported once the window containing it is complete.
    lookahead_ms = 32.0

    def __init__(self, settings):
        self.settings = settings
        self.model = load_silero_vad()

    def detect_segments(self, audio, sample_rate):
        vad_audio = resample_audio(audio, sample_rate, self.settings.target_sample_rate)
        vad_tensor = torch.from_numpy(vad_audio)
        segments = get_speech_timestamps(
            vad_tensor,
            self.model,
            sampling_rate=self.settings.target_sample_rate,
            threshold=self.settings.threshold,
            min_silence_duration_ms=self.settings.min_silence_duration_ms,
            speech_pad_ms=self.settings.speech_pad_ms,
            max_speech_duration_s=self.settings.max_speech_duration_s,
        )
        return vad_audio, self.settings.target_sample_rate, segments


class WebrtcVadRunner(BaseVadRunner):
    """GMM VAD from WebRTC. Frame-synchronous, so its lookahead is one frame."""

    name = "webrtc"
    SUPPORTED_FRAME_MS = (10, 20, 30)
    SUPPORTED_SAMPLE_RATES = (8000, 16000, 32000, 48000)

    def __init__(self, settings, aggressiveness=2, frame_duration_ms=30):
        import webrtcvad

        if frame_duration_ms not in self.SUPPORTED_FRAME_MS:
            raise ValueError("webrtc frame_duration_ms must be one of %s" % (self.SUPPORTED_FRAME_MS,))
        if settings.target_sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError("webrtc target_sample_rate must be one of %s" % (self.SUPPORTED_SAMPLE_RATES,))

        self.settings = settings
        self.aggressiveness = int(aggressiveness)
        self.frame_duration_ms = int(frame_duration_ms)
        self.lookahead_ms = float(frame_duration_ms)
        self.vad = webrtcvad.Vad(self.aggressiveness)

    def detect_segments(self, audio, sample_rate):
        target_rate = self.settings.target_sample_rate
        vad_audio = resample_audio(audio, sample_rate, target_rate)
        frame_length = int(target_rate * self.frame_duration_ms / 1000)
        if frame_length <= 0 or vad_audio.size < frame_length:
            return vad_audio, target_rate, []

        pcm = np.clip(vad_audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)

        frame_count = pcm.size // frame_length
        flags = []
        for index in range(frame_count):
            frame = pcm[index * frame_length : (index + 1) * frame_length]
            flags.append(self.vad.is_speech(frame.tobytes(), target_rate))

        segments = segments_from_frame_flags(
            flags=flags,
            frame_length_samples=frame_length,
            audio_length=vad_audio.size,
            min_silence_duration_ms=self.settings.min_silence_duration_ms,
            speech_pad_ms=self.settings.speech_pad_ms,
            max_speech_duration_s=self.settings.max_speech_duration_s,
            sample_rate=target_rate,
        )
        return vad_audio, target_rate, segments


class PyannoteVadRunner(BaseVadRunner):
    """pyannote segmentation-3.0, read as a voice-activity curve.

    The checkpoint is gated on the Hugging Face Hub: accept the model's
    conditions with an account, then expose a read token as HF_TOKEN (or pass
    auth_token in conf) or this raises at construction time.
    """

    name = "pyannote"
    DEFAULT_MODEL_ID = "pyannote/segmentation-3.0"

    def __init__(self, settings, model_id=None, auth_token=None, onset=0.5, device="cpu"):
        import os

        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection

        self.settings = settings
        self.onset = float(onset)
        token = auth_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            # Falls back to whatever `hf auth login` stored, so the token never
            # has to live in the repo or in a shell profile.
            from huggingface_hub import get_token

            token = get_token()
        if not token:
            raise RuntimeError(
                "pyannote/segmentation-3.0 is gated. Accept its conditions on the Hub, "
                "then run `hf auth login` (or set HF_TOKEN)."
            )
        model = Model.from_pretrained(model_id or self.DEFAULT_MODEL_ID, use_auth_token=token)
        # segmentation-3.0 scores 10s windows, so a boundary is not final until
        # the window covering it has been scored.
        self.lookahead_ms = float(getattr(model, "specifications", None) and model.specifications.duration * 1000.0 or 0.0)

        self.pipeline = VoiceActivityDetection(segmentation=model)
        self.pipeline.instantiate(
            {
                "min_duration_on": 0.0,
                "min_duration_off": self.settings.min_silence_duration_ms / 1000.0,
            }
        )
        if device and device != "cpu":
            self.pipeline.to(torch.device(device))

    def detect_segments(self, audio, sample_rate):
        target_rate = self.settings.target_sample_rate
        vad_audio = resample_audio(audio, sample_rate, target_rate)
        waveform = torch.from_numpy(vad_audio).unsqueeze(0)
        annotation = self.pipeline({"waveform": waveform, "sample_rate": target_rate})

        pad_samples = max(0, int(round((self.settings.speech_pad_ms / 1000.0) * target_rate)))
        segments = []
        for speech in annotation.get_timeline().support():
            start = max(0, int(round(speech.start * target_rate)) - pad_samples)
            end = min(vad_audio.size, int(round(speech.end * target_rate)) + pad_samples)
            if end > start:
                segments.append({"start": start, "end": end})
        return vad_audio, target_rate, segments


VAD_REGISTRY = {
    "silero": SileroVadRunner,
    "webrtc": WebrtcVadRunner,
    "pyannote": PyannoteVadRunner,
}


def create_vad_settings(config):
    return SileroVadSettings(
        threshold=float(config["silero_vad"]["threshold"]),
        min_silence_duration_ms=int(config["silero_vad"]["min_silence_duration_ms"]),
        speech_pad_ms=int(config["silero_vad"]["speech_pad_ms"]),
        max_speech_duration_s=float(config["silero_vad"]["max_speech_duration_s"]),
        target_sample_rate=int(config["silero_vad"]["target_sample_rate"]),
    )


def create_silero_vad_runner(config):
    """Unchanged entry point used by the baseline Step 2 run."""
    return SileroVadRunner(create_vad_settings(config))


def create_vad_runner(config, vad_name, overrides=None):
    """Build any registered VAD against the shared segmentation settings.

    The threshold / min_silence / speech_pad / max_speech values come from one
    place for every detector, so a difference between two VADs in E4 is a
    difference between the detectors.
    """
    if vad_name not in VAD_REGISTRY:
        raise ValueError("Unsupported VAD: %s. Supported: %s" % (vad_name, ", ".join(sorted(VAD_REGISTRY))))

    settings = create_vad_settings(config)
    if overrides:
        for key, value in overrides.items():
            if hasattr(settings, key):
                setattr(settings, key, value)

    extra = dict((config.get("vad", {}) or {}).get(vad_name, {}) or {})
    return VAD_REGISTRY[vad_name](settings, **extra)
