import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.io import read_audio, slice_audio, write_audio
from src.audio.oracle_segments import (
    GeneratedVariant,
    format_margin_pair,
    format_signed_ms,
    generate_pause_variants,
    margin_window,
)
from src.common import ensure_dir, load_config, read_jsonl, setup_logging, write_jsonl
from src.data.textgrid_parser import parse_textgrid


LOGGER = logging.getLogger("generate_oracle_variants")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate oracle segmentation variants from MFA TextGrids.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--logging-config", default=str(PROJECT_ROOT / "conf" / "logging.yaml"))
    parser.add_argument("--manifest", required=True, help="Input split manifest JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional utterance limit.")
    parser.add_argument(
        "--grid",
        choices=["axes", "grid2d", "both"],
        default="axes",
        help="axes: the baseline 1D sweeps (oracle.leading_ms / oracle.trailing_ms). "
        "grid2d: the leading x trailing product from margin_grid. "
        "both: their union, each window emitted once.",
    )
    parser.add_argument(
        "--materialize",
        choices=["wav", "offsets"],
        default="wav",
        help="wav: write one WAV per condition (the original behavior). "
        "offsets: record the window into the source file instead, so the "
        "decoder slices in memory and the grid costs no disk.",
    )
    parser.add_argument(
        "--output-suffix",
        default="oracle_variants",
        help="Variant manifest is written as <split>_<suffix>.jsonl. Change it so a "
        "2D grid run does not overwrite the reproduction run's manifest.",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Skip the pause-cut variants. They are not part of the margin grid, "
        "and unlike margin windows they splice two slices so they cannot be "
        "expressed as offsets.",
    )
    return parser.parse_args()


def build_margin_conditions(config, grid_mode):
    """The (leading_ms, trailing_ms) pairs to generate, each at most once.

    Pairs on an axis keep the baseline naming ("leading_p400", experiment_type
    "leading") because src/analysis/render_step1_views.py parses those. Off-axis
    pairs are new and get their own experiment_type, so the reproduction
    analysis ignores them and the 2D analysis can read both.
    """
    conditions = {}

    def add(leading_ms, trailing_ms):
        key = (int(leading_ms), int(trailing_ms))
        if key in conditions:
            return
        leading_ms, trailing_ms = key
        if trailing_ms == 0 and leading_ms != 0:
            conditions[key] = ("leading", format_signed_ms("leading", leading_ms))
        elif leading_ms == 0 and trailing_ms != 0:
            conditions[key] = ("trailing", format_signed_ms("trailing", trailing_ms))
        elif leading_ms == 0 and trailing_ms == 0:
            # The exact-boundary condition is one window but both 1D sweeps
            # report it. Keeping it under "leading" matches the original run.
            conditions[key] = ("leading", format_signed_ms("leading", 0))
        else:
            conditions[key] = ("margin_grid", format_margin_pair(leading_ms, trailing_ms))

    if grid_mode in ("axes", "both"):
        for value in config["oracle"]["leading_ms"]:
            add(value, 0)
        for value in config["oracle"]["trailing_ms"]:
            add(0, value)

    if grid_mode in ("grid2d", "both"):
        grid = config["margin_grid"]
        for leading_ms in grid["leading_ms"]:
            for trailing_ms in grid["trailing_ms"]:
                add(leading_ms, trailing_ms)

    return [(key, value[0], value[1]) for key, value in conditions.items()]


def build_variant_record(
    generated,
    source_row,
    output_audio_path,
):
    record = generated.to_dict()
    record.update(
        {
            "variant_id": generated.variant_id,
            "source_utterance_id": source_row["utterance_id"],
            "split": source_row["split"],
            "source_audio_path": source_row["audio_path"],
            "textgrid_path": source_row["textgrid_path"],
            "reference": source_row["transcript"],
        }
    )
    # Offset variants have no file of their own; the decoder reads the window
    # named by slice_start_sec/slice_end_sec out of source_audio_path.
    if output_audio_path is not None:
        record["generated_audio_path"] = str(output_audio_path.resolve())
    return record


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(args.logging_config)
    LOGGER.info(
        "START generate_oracle_variants | manifest=%s limit=%s grid=%s materialize=%s",
        args.manifest,
        args.limit,
        args.grid,
        args.materialize,
    )

    source_manifest_path = Path(args.manifest).resolve()
    records = read_jsonl(source_manifest_path)
    limit = args.limit
    if limit is not None:
        records = records[:limit]

    output_root = ensure_dir(config["oracle"]["audio_output_path"])
    variant_manifest_dir = ensure_dir(config["manifest"]["output_path"])
    min_variant_duration_sec = float(config["oracle"]["min_variant_duration_sec"])

    margin_conditions = build_margin_conditions(config, args.grid)
    LOGGER.info("Margin conditions: %d (grid=%s)", len(margin_conditions), args.grid)

    variant_records: list[dict] = []
    experiment_counter: Counter[str] = Counter()
    skipped_counter: Counter[str] = Counter()

    for row in records:
        utterance_id = row["utterance_id"]
        split = row["split"]
        split_output_root = ensure_dir(output_root / split)

        try:
            audio, sample_rate = read_audio(row["audio_path"])
            alignment = parse_textgrid(row["textgrid_path"])
        except Exception:
            LOGGER.exception("Failed to prepare source utterance: %s", utterance_id)
            skipped_counter["source_load_error"] += 1
            continue

        duration_sec = len(audio) / sample_rate

        for (leading_ms, trailing_ms), experiment_type, condition in margin_conditions:
            start_sec, end_sec = margin_window(alignment, duration_sec, leading_ms, trailing_ms)
            variant_duration_sec = max(0.0, end_sec - start_sec)
            if variant_duration_sec < min_variant_duration_sec:
                skipped_counter[f"{experiment_type}_too_short"] += 1
                continue

            variant_id = f"{utterance_id}__{condition}"
            output_path = None
            if args.materialize == "wav":
                variant_audio = slice_audio(audio, sample_rate, start_sec, end_sec)
                output_path = ensure_dir(split_output_root / experiment_type) / f"{variant_id}.wav"
                write_audio(output_path, variant_audio, sample_rate)
                variant_duration_sec = len(variant_audio) / sample_rate

            generated = GeneratedVariant(
                variant_id=variant_id,
                source_utterance_id=utterance_id,
                split=split,
                experiment_type=experiment_type,
                condition=condition,
                # Kept scalar for the 1D reproduction analysis, which plots one
                # axis at a time; leading_ms/trailing_ms below carry the pair.
                parameter_value_ms=trailing_ms if experiment_type == "trailing" else leading_ms,
                pause_bucket=None,
                pause_cut_ratio=None,
                pause_index=None,
                generated_duration_sec=variant_duration_sec,
                sample_rate=sample_rate,
                transcript_reference=row["transcript"],
                metadata={
                    "leading_ms": leading_ms,
                    "trailing_ms": trailing_ms,
                    "slice_start_sec": start_sec,
                    "slice_end_sec": end_sec,
                },
            )
            variant_records.append(build_variant_record(generated, row, output_path))
            experiment_counter[experiment_type] += 1

        if args.no_pause:
            continue

        pause_dir = ensure_dir(split_output_root / "pause_cut")

        pause_variants = generate_pause_variants(
            audio=audio,
            sample_rate=sample_rate,
            alignment=alignment,
            pause_buckets_ms=config["oracle"]["pause_buckets_ms"],
            pause_cut_positions=config["oracle"]["pause_cut_positions"],
            pause_min_duration_ms=config["oracle"]["pause_min_duration_ms"],
        )

        for generated, variant_audio in pause_variants:
            if generated.generated_duration_sec < min_variant_duration_sec:
                skipped_counter["pause_too_short"] += 1
                continue
            variant_id = f"{utterance_id}__{generated.condition}_i{generated.pause_index:02d}"
            output_path = pause_dir / f"{variant_id}.wav"
            write_audio(output_path, variant_audio, sample_rate)
            generated.variant_id = variant_id
            generated.source_utterance_id = utterance_id
            generated.split = split
            generated.transcript_reference = row["transcript"]
            variant_records.append(build_variant_record(generated, row, output_path))
            experiment_counter["pause_cut"] += 1

    output_manifest_path = variant_manifest_dir / f"{source_manifest_path.stem}_{args.output_suffix}.jsonl"
    write_jsonl(output_manifest_path, variant_records)
    LOGGER.info("Variant manifest written to %s", output_manifest_path)
    LOGGER.info("Generated counts: %s", dict(experiment_counter))
    LOGGER.info("Skipped counts: %s", dict(skipped_counter))
    LOGGER.info("END generate_oracle_variants | manifest=%s variants=%d", source_manifest_path, len(variant_records))


if __name__ == "__main__":
    main()
