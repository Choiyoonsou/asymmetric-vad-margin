from dataclasses import dataclass

import numpy as np

from src.common import get_split_group


@dataclass
class VadPolicySettings:
    policy_name: str
    leading_pad_ms: int
    trailing_pad_ms: int
    # Digital silence appended after the last segment (not waited for).
    trailing_silence_ms: int = 0


def milliseconds_to_samples(value_ms, sample_rate):
    return max(0, int(round((value_ms / 1000.0) * sample_rate)))


def normalize_segments(segments, audio_length):
    normalized = []
    for segment in segments:
        start = max(0, int(segment["start"]))
        end = min(audio_length, int(segment["end"]))
        if end <= start:
            continue
        normalized.append({"start": start, "end": end})
    normalized.sort(key=lambda segment: segment["start"])
    return normalized


def apply_vad_policy(segments, sample_rate, audio_length, settings):
    base_segments = normalize_segments(segments, audio_length)
    if not base_segments:
        return []

    leading_pad_samples = milliseconds_to_samples(settings.leading_pad_ms, sample_rate)
    trailing_pad_samples = milliseconds_to_samples(settings.trailing_pad_ms, sample_rate)

    adjusted = []
    for index, segment in enumerate(base_segments):
        start = max(0, segment["start"] - leading_pad_samples)
        end = min(audio_length, segment["end"] + trailing_pad_samples)

        if index < len(base_segments) - 1:
            next_segment = base_segments[index + 1]
            next_start_with_leading = max(0, next_segment["start"] - leading_pad_samples)
            if end > next_start_with_leading:
                end = next_start_with_leading

        if end <= start:
            continue
        adjusted.append(
            {
                "start": start,
                "end": end,
            }
        )

    return normalize_segments(adjusted, audio_length)


def create_vad_policy_settings(config):
    section = config["vad_policy"]
    return {
        "policy_name": section["policy_name"],
        "clean": VadPolicySettings(
            policy_name=section["policy_name"],
            leading_pad_ms=int(section["clean"]["leading_pad_ms"]),
            trailing_pad_ms=int(section["clean"]["trailing_pad_ms"]),
        ),
        "other": VadPolicySettings(
            policy_name=section["policy_name"],
            leading_pad_ms=int(section["other"]["leading_pad_ms"]),
            trailing_pad_ms=int(section["other"]["trailing_pad_ms"]),
        ),
    }


def get_vad_policy_settings(policy_config, split_name):
    group = get_split_group(split_name)
    return policy_config[group]


def resolve_policy(config, policy_name, split_name):
    """Settings for one named policy, or for the baseline split-group policy.

    "fa_informed" keeps the baseline behavior (clean 400/400, other 500/500,
    chosen by split group). Everything listed under vad_policies is a flat
    leading/trailing pair that applies to every split -- the asymmetric
    policies are not supposed to vary by split, since the claim is about which
    SIDE of the boundary the margin is spent on.
    """
    if policy_name in ("fa_informed", config["vad_policy"]["policy_name"]):
        return get_vad_policy_settings(create_vad_policy_settings(config), split_name)

    policies = config.get("vad_policies", {}) or {}
    if policy_name not in policies:
        known = ", ".join(sorted(set(policies) | {"fa_informed"}))
        raise ValueError("Unknown VAD policy: %s. Known: %s" % (policy_name, known))

    entry = policies[policy_name]
    return VadPolicySettings(
        policy_name=policy_name,
        leading_pad_ms=int(entry["leading_pad_ms"]),
        trailing_pad_ms=int(entry["trailing_pad_ms"]),
        trailing_silence_ms=int(entry.get("trailing_silence_ms", 0)),
    )


def append_trailing_silence(audio, sample_rate, settings):
    """Pad the end of the reconstructed utterance with digital silence.

    A streaming encoder with a short look-ahead needs frames after the last
    phone to emit the final token, but not real audio: silence recovers the
    same accuracy as waiting for the recording's tail, and it can be generated
    the moment the endpoint fires, so it adds no algorithmic latency.
    """
    silence = milliseconds_to_samples(settings.trailing_silence_ms, sample_rate)
    if silence == 0 or audio.size == 0:
        return audio
    return np.concatenate([audio, np.zeros(silence, dtype=audio.dtype)])


def algorithmic_latency_ms(config, settings, lookahead_ms, speech_pad_ms=None):
    """End-of-utterance latency this policy commits to, in milliseconds.

        min_silence_duration_ms + speech_pad_ms + trailing_pad_ms + VAD lookahead

    leading_pad_ms is deliberately absent. Widening the front of a segment
    re-reads audio that is already buffered, so it costs nothing to wait for --
    that asymmetry is the whole hypothesis, and this function is where it is
    made explicit rather than argued in prose.

    speech_pad_ms is the detector's own symmetric pad, applied before the
    policy's. It extends the tail too, so it is waited for like trailing_pad_ms;
    leaving it out recorded every E2 speech_pad condition at the same latency.
    """
    if speech_pad_ms is None:
        speech_pad_ms = config["silero_vad"]["speech_pad_ms"]
    return (
        float(config["silero_vad"]["min_silence_duration_ms"])
        + float(speech_pad_ms)
        + float(settings.trailing_pad_ms)
        + float(lookahead_ms)
    )
