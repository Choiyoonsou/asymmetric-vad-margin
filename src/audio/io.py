from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio


RAW_PCM_SAMPLE_RATE = 16000
RAW_PCM_SUFFIXES = {".pcm"}


def read_raw_pcm(path, sample_rate=RAW_PCM_SAMPLE_RATE):
    # Headerless 16-bit signed little-endian mono PCM, as shipped by some
    # KsponSpeech distributions. Adjust sample_rate if your copy differs.
    raw = np.fromfile(str(path), dtype=np.int16)
    audio = (raw.astype(np.float32)) / 32768.0
    return audio, sample_rate


def read_audio(path):
    path = Path(path)
    if path.suffix.lower() in RAW_PCM_SUFFIXES:
        return read_raw_pcm(path)

    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio = waveform.squeeze(0).numpy().astype(np.float32)
    return audio, sample_rate


def write_audio(path, audio, sample_rate):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    torchaudio.save(str(output_path), waveform, sample_rate)


def get_audio_duration(path):
    path = Path(path)
    if path.suffix.lower() in RAW_PCM_SUFFIXES:
        audio, sample_rate = read_raw_pcm(path)
        return float(len(audio)) / float(sample_rate)

    waveform, sample_rate = torchaudio.load(str(path))
    return float(waveform.shape[-1]) / float(sample_rate)


def seconds_to_samples(time_sec, sample_rate):
    return max(0, int(round(time_sec * sample_rate)))


def clamp_time(time_sec, minimum, maximum):
    return max(minimum, min(time_sec, maximum))


def slice_audio(audio, sample_rate, start_sec, end_sec):
    start_index = seconds_to_samples(start_sec, sample_rate)
    end_index = seconds_to_samples(end_sec, sample_rate)
    if end_index <= start_index:
        return np.array([], dtype=np.float32)
    return audio[start_index:end_index].copy()


def concatenate_segments(segments):
    valid_segments = [segment for segment in segments if segment.size > 0]
    if not valid_segments:
        return np.array([], dtype=np.float32)
    return np.concatenate(valid_segments).astype(np.float32)


@dataclass(frozen=True)
class AudioSource:
    """What to decode: a whole file, or a window inside one.

    The margin grid multiplies every utterance by dozens of conditions. Writing
    each condition out as its own WAV costs ~120GB for the full grid and makes
    disk I/O, not the GPU, the bottleneck. A window carries the same
    information as the clip, so a variant manifest records offsets instead and
    the decoder slices in memory.
    """

    path: str
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None

    @property
    def is_window(self):
        return self.start_sec is not None or self.end_sec is not None


@dataclass(frozen=True)
class AudioArray:
    """Audio that already exists in memory.

    VAD decoding reconstructs a speech-only waveform per utterance per policy.
    Writing each one out would cost hundreds of thousands of files across the
    E2 sweep and the E4 grid, and nothing downstream reads them again.
    """

    samples: object
    sample_rate: int


def load_audio_source(source):
    """Read an AudioSource, an AudioArray, or a plain path into (audio, sample_rate)."""
    if isinstance(source, AudioArray):
        return source.samples, source.sample_rate
    if not isinstance(source, AudioSource):
        return read_audio(source)

    audio, sample_rate = read_audio(source.path)
    if not source.is_window:
        return audio, sample_rate

    duration_sec = len(audio) / sample_rate
    start_sec = 0.0 if source.start_sec is None else clamp_time(source.start_sec, 0.0, duration_sec)
    end_sec = duration_sec if source.end_sec is None else clamp_time(source.end_sec, 0.0, duration_sec)
    return slice_audio(audio, sample_rate, start_sec, end_sec), sample_rate
