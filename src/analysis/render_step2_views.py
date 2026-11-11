import argparse
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.common import ensure_dir, get_split_group, load_config, setup_logging


LOGGER = logging.getLogger("render_step2_views")


def parse_args():
    parser = argparse.ArgumentParser(description="Render the Step 2 table.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--logging-config", default=str(PROJECT_ROOT / "conf" / "logging.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exp-subdir", default="", help="Optional experiment subdirectory under exp_path, e.g. step2.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def condition_label(condition, policy_tag):
    if condition == "baseline":
        return "baseline"
    if condition == "silero_vad_default":
        return "vad_default"
    if condition == policy_tag:
        return "vad_fa_informed"
    return condition


def format_percent(series):
    return (series * 100.0).round(2)


def build_cer_boundary_changed_table(frame):
    group_columns = ["split_group", "condition_label"]
    summary = (
        frame.groupby(group_columns, dropna=False)[
            [
                "cer",
                "first_word_error_rate",
                "last_word_error_rate",
                "changed_from_baseline",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    counts = frame.groupby(group_columns, dropna=False).size().reset_index(name="n_examples")
    summary = summary.merge(counts, on=group_columns, how="left")
    summary["cer_mean_pct"] = format_percent(summary["cer"])
    summary["first_word_error_rate_pct"] = format_percent(summary["first_word_error_rate"])
    summary["last_word_error_rate_pct"] = format_percent(summary["last_word_error_rate"])
    summary["changed_rate_pct"] = format_percent(summary["changed_from_baseline"])

    table = summary[
        [
            "split_group",
            "condition_label",
            "n_examples",
            "cer_mean_pct",
            "first_word_error_rate_pct",
            "last_word_error_rate_pct",
            "changed_rate_pct",
        ]
    ].copy()
    return table.sort_values(["split_group", "condition_label"]).reset_index(drop=True)


def main():
    args = parse_args()
    config = load_config(args.config)
    setup_logging(args.logging_config)
    LOGGER.info("START render_step2_views | dataset=%s exp_subdir=%s output_dir=%s", args.dataset, args.exp_subdir, args.output_dir)

    exp_root = Path(config["exp_path"])
    if args.exp_subdir:
        exp_root = exp_root / args.exp_subdir
    output_dir = Path(args.output_dir) if args.output_dir else exp_root / "analysis"
    tables_dir = ensure_dir(output_dir / "tables")

    eval_dir = exp_root / "eval"
    paths = sorted(eval_dir.glob(f"{args.dataset}_*_per_utterance.csv"))
    if not paths:
        raise FileNotFoundError(f"No step2 per-utterance CSV files found for dataset={args.dataset}")

    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    policy_tag = f"silero_vad_{config['vad_policy']['policy_name']}"
    valid_conditions = {"baseline", "silero_vad_default", policy_tag}
    frame = frame[frame["condition"].isin(valid_conditions)].copy()
    frame["split_group"] = frame["split"].map(get_split_group)
    frame["condition_label"] = frame["condition"].map(lambda value: condition_label(value, policy_tag))

    table = build_cer_boundary_changed_table(frame)
    output_path = tables_dir / "cer_boundary_changed_table.csv"
    table.to_csv(output_path, index=False)
    LOGGER.info("Wrote table %s", output_path)
    LOGGER.info("END render_step2_views | dataset=%s exp_subdir=%s output_dir=%s", args.dataset, args.exp_subdir, output_dir)


if __name__ == "__main__":
    main()
