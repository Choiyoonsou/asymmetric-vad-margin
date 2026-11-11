import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import ensure_dir, load_config, read_jsonl, set_cuda_visible_devices, setup_logging, write_jsonl


LOGGER = logging.getLogger("run_silero_vad_decode")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Silero VAD on source audio, reconstruct speech-only audio, and decode it with wav2vec2-base-960h.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--logging-config", default=str(PROJECT_ROOT / "conf" / "logging.yaml"))
    parser.add_argument("--manifest", required=True, help="Baseline split manifest JSONL.")
    parser.add_argument("--policy", required=True, choices=["default", "fa_informed"], help="VAD policy mode.")
    parser.add_argument("--exp-subdir", default="", help="Optional experiment subdirectory under exp_path, e.g. step2.")
    parser.add_argument("--output", default=None, help="Optional decode JSONL output path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional decode limit.")
    parser.add_argument(
        "--model",
        default=None,
        help="ASR backbone for this run. Overrides conf asr.model_name; see src/asr.py:BACKBONE_REGISTRY.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Decoding language for this run. Overrides conf asr.language; pass 'none' to clear it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_cuda_visible_devices(config)
    from src.audio.io import read_audio, write_audio
    from src.asr import apply_asr_overrides, create_asr_runners, get_asr_model_tag
    from src.vad import create_silero_vad_runner
    from src.vad_policy import apply_vad_policy, create_vad_policy_settings, get_vad_policy_settings

    apply_asr_overrides(config, args.model, args.language)
    setup_logging(args.logging_config)
    LOGGER.info(
        "START run_silero_vad_decode | manifest=%s policy=%s exp_subdir=%s limit=%s output=%s model=%s",
        args.manifest,
        args.policy,
        args.exp_subdir,
        args.limit,
        args.output,
        config["asr"]["model_name"],
    )

    input_manifest = Path(args.manifest).resolve()
    rows = read_jsonl(input_manifest)
    limit = args.limit
    if limit is not None:
        rows = rows[:limit]

    if args.policy == "default":
        policy_tag = "silero_vad_default"
        policy_name = "default"
    else:
        policy_config = create_vad_policy_settings(config)
        policy_name = policy_config["policy_name"]
        policy_tag = f"silero_vad_{policy_name}"

    exp_root = Path(config["exp_path"])
    if args.exp_subdir:
        exp_root = exp_root / args.exp_subdir
    output_dir = ensure_dir(exp_root / "decodes")
    vad_audio_root = ensure_dir(Path(config["artifacts_path"]) / "generated_audio" / policy_tag)
    output_path = (
        Path(args.output).resolve()
        if args.output
        else output_dir / f"{input_manifest.stem}__{policy_tag}__{get_asr_model_tag(config)}.jsonl"
    )

    asr_runners = create_asr_runners(config)
    num_workers = min(int(config["asr"]["num_workers"]), len(asr_runners))

    def process_rows(worker_rows, asr_runner):
        vad_runner = create_silero_vad_runner(config)
        worker_results = []
        for row in worker_rows:
            utterance_id = row["utterance_id"]
            LOGGER.info("Running Silero VAD on GPU %s: %s", asr_runner.settings.device_index, utterance_id)
            try:
                audio, sample_rate = read_audio(row["audio_path"])
                vad_audio, vad_sample_rate, segments = vad_runner.detect_segments(audio, sample_rate)
                output_segments = segments
                leading_pad_ms = 0
                trailing_pad_ms = 0
                if args.policy == "fa_informed":
                    policy_settings = get_vad_policy_settings(policy_config, row["split"])
                    leading_pad_ms = policy_settings.leading_pad_ms
                    trailing_pad_ms = policy_settings.trailing_pad_ms
                    output_segments = apply_vad_policy(
                        segments=segments,
                        sample_rate=vad_sample_rate,
                        audio_length=len(vad_audio),
                        settings=policy_settings,
                    )
                reconstructed = vad_runner.reconstruct_audio(vad_audio, output_segments)
            except Exception:
                LOGGER.exception("Failed during VAD preprocessing: %s", utterance_id)
                continue

            if reconstructed.size == 0:
                LOGGER.warning("Skipping empty VAD reconstruction: %s", utterance_id)
                continue

            split_output_root = ensure_dir(vad_audio_root / row["split"])
            output_audio_path = split_output_root / f"{utterance_id}__{policy_tag}.wav"
            write_audio(output_audio_path, reconstructed, vad_sample_rate)

            decode_result = asr_runner.transcribe(output_audio_path)
            worker_results.append(
                {
                    "variant_id": f"{utterance_id}__{policy_tag}",
                    "source_utterance_id": utterance_id,
                    "experiment_type": "silero_vad",
                    "condition": policy_tag,
                    "split": row["split"],
                    "hypothesis": decode_result["hypothesis"],
                    "reference": row["transcript"],
                    "audio_path": str(output_audio_path.resolve()),
                    "segment_count": decode_result["segment_count"],
                    "vad_segment_count": len(segments),
                    "vad_output_segment_count": len(output_segments),
                    "vad_speech_duration_sec": len(reconstructed) / vad_sample_rate,
                    "vad_policy_name": policy_name,
                    "vad_threshold": config["silero_vad"]["threshold"],
                    "vad_min_silence_duration_ms": config["silero_vad"]["min_silence_duration_ms"],
                    "vad_speech_pad_ms": config["silero_vad"]["speech_pad_ms"],
                    "vad_policy_leading_pad_ms": leading_pad_ms,
                    "vad_policy_trailing_pad_ms": trailing_pad_ms,
                    "model_name": config["asr"]["model_name"],
                    "device": config["asr"]["device"],
                    "device_index": asr_runner.settings.device_index,
                    "physical_device_index": asr_runner.settings.physical_device_index,
                }
            )
        return worker_results

    results = []
    if num_workers > 1 and len(asr_runners) > 1:
        worker_count = min(num_workers, len(asr_runners))
        row_groups = []
        for group_index in range(worker_count):
            row_groups.append(rows[group_index::worker_count])

        futures = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for worker_index in range(worker_count):
                futures.append(executor.submit(process_rows, row_groups[worker_index], asr_runners[worker_index]))
            for future in futures:
                results.extend(future.result())
    else:
        results = process_rows(rows, asr_runners[0])

    write_jsonl(output_path, results)
    LOGGER.info("Silero VAD decode output written to %s", output_path)
    LOGGER.info(
        "END run_silero_vad_decode | manifest=%s policy=%s exp_subdir=%s rows=%d output=%s",
        input_manifest,
        args.policy,
        args.exp_subdir,
        len(results),
        output_path,
    )


if __name__ == "__main__":
    main()
