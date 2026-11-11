"""Decode a split through a VAD + margin policy.

Generalizes scripts/run_silero_vad_decode.py along the two axes the follow-up
study adds: which detector runs (Silero / WebRTC / pyannote) and which named
margin policy pads its segments. The original script is left untouched so the
baseline Step 2 run keeps running exactly the baseline path.

    # E2 -- is the policy anything more than a wider speech_pad?
    python scripts/run_vad_decode.py --manifest ... --vad silero --policy default --speech-pad-ms 400

    # E4 -- same policy across three detectors
    python scripts/run_vad_decode.py --manifest ... --vad webrtc --policy asym_400_050
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import ensure_dir, load_config, read_jsonl, set_cuda_visible_devices, setup_logging, write_jsonl


LOGGER = logging.getLogger("run_vad_decode")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--logging-config", default=str(PROJECT_ROOT / "conf" / "logging.yaml"))
    parser.add_argument("--manifest", required=True, help="Baseline split manifest JSONL.")
    parser.add_argument("--vad", default="silero", help="Detector name; see src/vad.py:VAD_REGISTRY.")
    parser.add_argument(
        "--policy",
        default="default",
        help="Margin policy: default, fa_informed, or any key under conf vad_policies.",
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=None,
        help="Override the detector's own speech_pad_ms. This is the E2 control: "
        "sweeping it widens both sides symmetrically without any policy logic.",
    )
    parser.add_argument(
        "--leading-pad-ms",
        type=int,
        default=None,
        help="Bypass --policy and set leading_pad_ms directly, for a grid sweep without enumerating "
        "named policies in YAML. Requires --trailing-pad-ms too.",
    )
    parser.add_argument(
        "--trailing-pad-ms",
        type=int,
        default=None,
        help="Bypass --policy and set trailing_pad_ms directly. See --leading-pad-ms.",
    )
    parser.add_argument(
        "--cap-vad-only",
        action="store_true",
        help="Decomposition control for --context-cap-ms. Trim the clip for the DETECTOR only, "
        "then apply the margin policy against the FULL clip. Capping normally changes two things "
        "at once -- what the VAD sees, and how much room the padding has. With this flag the VAD "
        "still sees the short clip but the padding room is unlimited, so a surviving effect is "
        "the detector's segmentation and a vanishing one is the padding room.",
    )
    parser.add_argument(
        "--context-cap-ms",
        type=float,
        default=None,
        help="Trim the clip to at most this much audio outside the forced-alignment "
        "boundaries BEFORE the VAD sees it. This is the headroom control: AMI and "
        "TED-LIUM clips carry 1000ms of context while LibriSpeech carries a median "
        "of ~350ms, so capping reproduces LibriSpeech's condition on a dataset that "
        "otherwise has room. It trims what the detector sees, not merely what the "
        "policy can pad into -- which is what actually happens on LibriSpeech.",
    )
    parser.add_argument("--exp-subdir", default="", help="Experiment subdirectory under exp_path, e.g. step2.")
    parser.add_argument("--output", default=None, help="Optional decode JSONL output path.")
    parser.add_argument("--run-tag", default=None, help="Override the condition tag used in output names.")
    parser.add_argument("--limit", type=int, default=None, help="Optional decode limit.")
    parser.add_argument("--model", default=None, help="ASR backbone override; see src/asr.py:BACKBONE_REGISTRY.")
    parser.add_argument("--language", default=None, help="Decoding language override; 'none' clears it.")
    return parser.parse_args()


def build_run_tag(args):
    if args.run_tag:
        return args.run_tag
    if args.leading_pad_ms is not None:
        tag = f"{args.vad}_vad_grid_l{args.leading_pad_ms:03d}_t{args.trailing_pad_ms:03d}"
    else:
        tag = f"{args.vad}_vad_{args.policy}"
    if args.speech_pad_ms is not None:
        tag = f"{tag}_pad{args.speech_pad_ms:03d}"
    if args.context_cap_ms is not None:
        tag = f"{tag}_cap{int(args.context_cap_ms):04d}"
        if args.cap_vad_only:
            tag = f"{tag}_vadonly"
    return tag


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_cuda_visible_devices(config)
    from src.audio.io import AudioArray, read_audio, slice_audio
    from src.data.textgrid_parser import parse_textgrid
    from src.asr import apply_asr_overrides, create_asr_runners, get_asr_model_tag
    from src.vad import create_vad_runner, resample_audio
    from src.vad_policy import algorithmic_latency_ms, append_trailing_silence, apply_vad_policy, resolve_policy

    apply_asr_overrides(config, args.model, args.language)
    setup_logging(args.logging_config)

    if (args.leading_pad_ms is None) != (args.trailing_pad_ms is None):
        raise ValueError("--leading-pad-ms and --trailing-pad-ms must be given together.")
    grid_settings = None
    if args.leading_pad_ms is not None:
        from src.vad_policy import VadPolicySettings

        grid_settings = VadPolicySettings(
            policy_name=f"grid_l{args.leading_pad_ms}_t{args.trailing_pad_ms}",
            leading_pad_ms=args.leading_pad_ms,
            trailing_pad_ms=args.trailing_pad_ms,
        )

    run_tag = build_run_tag(args)
    overrides = {"speech_pad_ms": args.speech_pad_ms} if args.speech_pad_ms is not None else None

    LOGGER.info(
        "START run_vad_decode | manifest=%s vad=%s policy=%s speech_pad_ms=%s tag=%s model=%s",
        args.manifest,
        args.vad,
        args.policy,
        args.speech_pad_ms,
        run_tag,
        config["asr"]["model_name"],
    )

    input_manifest = Path(args.manifest).resolve()
    rows = read_jsonl(input_manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    exp_root = Path(config["exp_path"])
    if args.exp_subdir:
        exp_root = exp_root / args.exp_subdir
    output_dir = ensure_dir(exp_root / "decodes")
    output_path = (
        Path(args.output).resolve()
        if args.output
        else output_dir / f"{input_manifest.stem}__{run_tag}__{get_asr_model_tag(config)}.jsonl"
    )

    asr_runners = create_asr_runners(config)
    num_workers = min(int(config["asr"]["num_workers"]), len(asr_runners))

    def process_rows(worker_rows, asr_runner):
        vad_runner = create_vad_runner(config, args.vad, overrides)
        worker_results = []
        for row in worker_rows:
            utterance_id = row["utterance_id"]
            try:
                audio, sample_rate = read_audio(row["audio_path"])
                cap_start_sec = 0.0
                detect_audio = audio
                if args.context_cap_ms is not None:
                    alignment = parse_textgrid(row["textgrid_path"])
                    cap_sec = args.context_cap_ms / 1000.0
                    duration_sec = len(audio) / sample_rate
                    start_sec = max(0.0, min(alignment.utterance_start - cap_sec, duration_sec))
                    end_sec = max(start_sec, min(alignment.utterance_end + cap_sec, duration_sec))
                    detect_audio = slice_audio(audio, sample_rate, start_sec, end_sec)
                    cap_start_sec = start_sec
                    if not args.cap_vad_only:
                        audio = detect_audio
                        cap_start_sec = 0.0

                vad_audio, vad_sample_rate, segments = vad_runner.detect_segments(detect_audio, sample_rate)

                if args.cap_vad_only and args.context_cap_ms is not None:
                    # Segments are in capped-clip coordinates; shift them back into the
                    # full clip so the policy can pad into audio the detector never saw.
                    vad_audio = resample_audio(audio, sample_rate, vad_sample_rate)
                    offset = int(round(cap_start_sec * vad_sample_rate))
                    segments = [
                        {"start": seg["start"] + offset, "end": seg["end"] + offset} for seg in segments
                    ]

                policy_settings = grid_settings if grid_settings is not None else resolve_policy(config, args.policy, row["split"])
                if policy_settings.leading_pad_ms == 0 and policy_settings.trailing_pad_ms == 0:
                    output_segments = segments
                else:
                    output_segments = apply_vad_policy(
                        segments=segments,
                        sample_rate=vad_sample_rate,
                        audio_length=len(vad_audio),
                        settings=policy_settings,
                    )
                reconstructed = vad_runner.reconstruct_audio(vad_audio, output_segments)
                reconstructed = append_trailing_silence(reconstructed, vad_sample_rate, policy_settings)
            except Exception:
                LOGGER.exception("Failed during VAD preprocessing: %s", utterance_id)
                continue

            if reconstructed.size == 0:
                # The VAD found no speech, so the system outputs nothing. Record
                # that as an empty hypothesis (CER 1.0) instead of dropping the
                # row, which silently excluded the hardest utterances from eval.
                LOGGER.warning("Empty VAD reconstruction, recording empty hypothesis: %s", utterance_id)
                decode_result = {"hypothesis": "", "duration_sec": 0.0, "segment_count": 0}
            else:
                decode_result = asr_runner.transcribe(AudioArray(reconstructed, vad_sample_rate))
            effective_speech_pad_ms = (
                args.speech_pad_ms if args.speech_pad_ms is not None else config["silero_vad"]["speech_pad_ms"]
            )
            worker_results.append(
                {
                    "variant_id": f"{utterance_id}__{run_tag}",
                    "source_utterance_id": utterance_id,
                    "experiment_type": f"{args.vad}_vad",
                    "condition": run_tag,
                    "split": row["split"],
                    "hypothesis": decode_result["hypothesis"],
                    "reference": row["transcript"],
                    "audio_path": row["audio_path"],
                    "decoded_duration_sec": decode_result["duration_sec"],
                    "segment_count": decode_result["segment_count"],
                    "vad_name": vad_runner.name,
                    "vad_segment_count": len(segments),
                    "vad_output_segment_count": len(output_segments),
                    "vad_speech_duration_sec": len(reconstructed) / vad_sample_rate,
                    "vad_lookahead_ms": vad_runner.lookahead_ms,
                    "vad_policy_name": policy_settings.policy_name,
                    "vad_policy_leading_pad_ms": policy_settings.leading_pad_ms,
                    "vad_policy_trailing_pad_ms": policy_settings.trailing_pad_ms,
                    "vad_policy_trailing_silence_ms": policy_settings.trailing_silence_ms,
                    "vad_speech_pad_ms": effective_speech_pad_ms,
                    "vad_min_silence_duration_ms": config["silero_vad"]["min_silence_duration_ms"],
                    "context_cap_ms": args.context_cap_ms,
                    "cap_vad_only": bool(args.cap_vad_only),
                    "algorithmic_latency_ms": algorithmic_latency_ms(
                        config, policy_settings, vad_runner.lookahead_ms, effective_speech_pad_ms
                    ),
                    "model_name": config["asr"]["model_name"],
                    "device": config["asr"]["device"],
                    "device_index": asr_runner.settings.device_index,
                    "physical_device_index": asr_runner.settings.physical_device_index,
                }
            )
        return worker_results

    results = []
    if num_workers > 1 and len(asr_runners) > 1:
        row_groups = [rows[index::num_workers] for index in range(num_workers)]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_rows, row_groups[index], asr_runners[index]) for index in range(num_workers)
            ]
            for future in futures:
                results.extend(future.result())
    else:
        results = process_rows(rows, asr_runners[0])

    write_jsonl(output_path, results)
    LOGGER.info(
        "END run_vad_decode | manifest=%s vad=%s policy=%s rows=%d output=%s",
        input_manifest,
        args.vad,
        args.policy,
        len(results),
        output_path,
    )


if __name__ == "__main__":
    main()
