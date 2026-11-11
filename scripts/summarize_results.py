import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import ensure_dir, load_config, read_jsonl, setup_logging
from src.audio.oracle_segments import parse_margin_condition
from src.metrics import compute_text_metrics, summarize_cer


LOGGER = logging.getLogger("summarize_results")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ASR decode results into CSV metrics.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--logging-config", default=str(PROJECT_ROOT / "conf" / "logging.yaml"))
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more decode JSONL files.")
    parser.add_argument("--exp-subdir", default="", help="Optional experiment subdirectory under exp_path, e.g. step1 or step2.")
    parser.add_argument("--output-prefix", default="summary", help="Output file prefix under exp/eval.")
    return parser.parse_args()


def attach_metrics(frame):
    metric_rows = [compute_text_metrics(row.reference, row.hypothesis) for row in frame.itertuples()]
    metric_rows = pd.DataFrame(metric_rows)
    return pd.concat([frame.reset_index(drop=True), metric_rows], axis=1)


def attach_baseline_change(frame):
    baseline = frame[frame["experiment_type"] == "baseline"][
        ["source_utterance_id", "split", "normalized_hypothesis"]
    ].drop_duplicates(subset=["source_utterance_id", "split"])
    if baseline.empty:
        LOGGER.info("No baseline rows found; skipping changed-vs-baseline summary.")
        frame["changed_from_baseline"] = pd.NA
        return frame

    baseline = baseline.rename(columns={"normalized_hypothesis": "baseline_hypothesis"})
    merged = frame.merge(baseline, on=["source_utterance_id", "split"], how="left")
    merged["changed_from_baseline"] = (
        merged["baseline_hypothesis"].notna()
        & (merged["normalized_hypothesis"] != merged["baseline_hypothesis"])
    )
    return merged


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(args.logging_config)
    LOGGER.info("START summarize_results | inputs=%s exp_subdir=%s output_prefix=%s", args.inputs, args.exp_subdir, args.output_prefix)

    all_rows: list[dict] = []
    for input_path_str in args.inputs:
        input_path = Path(input_path_str).resolve()
        all_rows.extend(read_jsonl(input_path))

    if not all_rows:
        raise ValueError("No decode rows found.")

    frame = pd.DataFrame(all_rows)
    frame = attach_metrics(frame)
    frame = attach_baseline_change(frame)

    exp_root = Path(config["exp_path"])
    if args.exp_subdir:
        exp_root = exp_root / args.exp_subdir
    eval_dir = ensure_dir(exp_root / "eval")
    per_utterance_path = eval_dir / f"{args.output_prefix}_per_utterance.csv"
    summary_path = eval_dir / f"{args.output_prefix}_summary.csv"

    frame.to_csv(per_utterance_path, index=False)

    summary_columns = [
        "cer",
        "exact_match_rate",
        "first_word_match_rate",
        "first_word_error_rate",
        "last_word_match_rate",
        "last_word_error_rate",
    ]
    if "changed_from_baseline" in frame.columns:
        summary_columns.append("changed_from_baseline")

    group_keys = ["experiment_type", "condition", "split"]
    summary = (
        frame.groupby(group_keys, dropna=False)[summary_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    counts = frame.groupby(group_keys, dropna=False).size().reset_index(name="n_examples")
    summary = summary.merge(counts, on=group_keys, how="left")

    # The mean alone hides the tail: a few runaway hypotheses can flip the sign
    # of a policy's apparent effect. Report median, capped mean, hallucination
    # rate and a bootstrap CI beside it so the comparison is falsifiable.
    robust_rows = []
    for keys, group in frame.groupby(group_keys, dropna=False):
        stats = summarize_cer(group["cer"].tolist())
        stats.pop("n_examples", None)
        robust_rows.append(dict(zip(group_keys, keys), **stats))
    summary = summary.merge(pd.DataFrame(robust_rows), on=group_keys, how="left")

    # The margin pair, so the 2D grid analysis can pivot without re-parsing.
    # Decode records written before the leading_ms/trailing_ms passthrough do
    # not carry the columns, but the condition name encodes the same pair.
    pairs = summary["condition"].map(parse_margin_condition)
    summary["leading_ms"] = [pair[0] if pair else pd.NA for pair in pairs]
    summary["trailing_ms"] = [pair[1] if pair else pd.NA for pair in pairs]
    summary.to_csv(summary_path, index=False)

    LOGGER.info("Per-utterance metrics written to %s", per_utterance_path)
    LOGGER.info("Grouped summary written to %s", summary_path)
    LOGGER.info("END summarize_results | exp_subdir=%s rows=%d output_prefix=%s", args.exp_subdir, len(frame), args.output_prefix)


if __name__ == "__main__":
    main()
