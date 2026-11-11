import argparse
import csv
import logging
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.common import ensure_dir, get_split_group, load_config, setup_logging


LOGGER = logging.getLogger("render_step1_views")

MAX_MARGIN_MS = 500


def parse_args():
    parser = argparse.ArgumentParser(description="Render the Step 1 tables and figures.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--logging-config", default=str(PROJECT_ROOT / "conf" / "logging.yaml"))
    parser.add_argument("--dataset", required=True, help="Dataset prefix used in exp/eval filenames.")
    parser.add_argument("--exp-subdir", default="", help="Optional experiment subdirectory under exp_path, e.g. step1.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Default: exp/<exp-subdir>/analysis")
    return parser.parse_args()


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value):
    if value in (None, "", "nan", "NaN"):
        return None
    return float(value)


def mean_value(values):
    if not values:
        return None
    return statistics.fmean(values)


def std_value(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def format_percent(value):
    if value is None:
        return ""
    return f"{value * 100.0:.2f}"


def load_per_utterance_rows(exp_root, dataset_name):
    paths = sorted((Path(exp_root) / "eval").glob(f"{dataset_name}_*_per_utterance.csv"))
    if not paths:
        raise FileNotFoundError(f"No per-utterance CSV files found for dataset={dataset_name}")

    rows = []
    for path in paths:
        rows.extend(read_csv_rows(path))
    return rows


def parse_margin_condition(condition):
    if not condition.startswith(("leading_", "trailing_")):
        return None
    suffix = condition.split("_", 1)[1]
    sign = suffix[0]
    value = int(suffix[1:])
    if sign == "m":
        return -value
    return value


def collect_margin_rows(raw_rows):
    rows = []
    for row in raw_rows:
        if row["experiment_type"] not in ("leading", "trailing"):
            continue

        margin_ms = parse_margin_condition(row["condition"])
        if margin_ms is None or margin_ms > MAX_MARGIN_MS:
            continue

        rows.append(
            {
                "split_group": get_split_group(row["split"]),
                "experiment_type": row["experiment_type"],
                "condition": row["condition"],
                "margin_ms": margin_ms,
                "cer": parse_float(row["cer"]),
                "first_word_error_rate": parse_float(row["first_word_error_rate"]),
                "last_word_error_rate": parse_float(row["last_word_error_rate"]),
            }
        )
    return rows


def aggregate_rows(rows, group_fields, metric_fields):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(row)

    output = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        record = {field: key[index] for index, field in enumerate(group_fields)}
        record["n_examples"] = len(group_rows)
        for metric in metric_fields:
            values = [row[metric] for row in group_rows if row[metric] is not None]
            record[f"{metric}_mean"] = mean_value(values)
            record[f"{metric}_std"] = std_value(values)
        output.append(record)
    return output


def build_leading_trailing_cer_table(rows):
    aggregated = aggregate_rows(
        rows,
        ["split_group", "experiment_type", "condition", "margin_ms"],
        ["cer"],
    )
    table = []
    for row in aggregated:
        table.append(
            {
                "split_group": row["split_group"],
                "experiment_type": row["experiment_type"],
                "margin_ms": row["margin_ms"],
                "condition": row["condition"],
                "cer_mean_pct": format_percent(row["cer_mean"]),
                "cer_std_pct": format_percent(row["cer_std"]),
            }
        )
    table.sort(key=lambda row: (row["split_group"], row["experiment_type"], float(row["margin_ms"])))
    return table


def build_first_and_last_word_errors_table(rows):
    aggregated = aggregate_rows(
        rows,
        ["experiment_type", "condition", "margin_ms"],
        ["first_word_error_rate", "last_word_error_rate"],
    )
    table = []
    for row in aggregated:
        table.append(
            {
                "experiment_type": row["experiment_type"],
                "condition": row["condition"],
                "margin_ms": row["margin_ms"],
                "n_examples": row["n_examples"],
                "first_word_error_rate_pct": format_percent(row["first_word_error_rate_mean"]),
                "first_word_error_rate_std_pct": format_percent(row["first_word_error_rate_std"]),
                "last_word_error_rate_pct": format_percent(row["last_word_error_rate_mean"]),
                "last_word_error_rate_std_pct": format_percent(row["last_word_error_rate_std"]),
            }
        )
    table.sort(key=lambda row: (row["experiment_type"], float(row["margin_ms"])))
    return table


def save_table(output_dir, name, rows):
    if not rows:
        return
    csv_path = output_dir / f"{name}.csv"
    write_csv(csv_path, list(rows[0].keys()), rows)
    LOGGER.info("Wrote table %s", csv_path)


def save_figure(fig, output_dir, name):
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote figure %s", pdf_path)


def format_margin_ticks(x_values):
    return [value for value in x_values if value % 100 == 0]


def render_leading_trailing_cer(rows, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    fig.subplots_adjust(top=0.82, wspace=0.24)
    panel_titles = {
        "clean": "Clean (dev+test)",
        "other": "Other (dev+test)",
    }

    for axis, split_group in zip(axes, ["clean", "other"]):
        x_ticks = []
        for experiment_type, color in [("leading", "#d1495b"), ("trailing", "#2b59c3")]:
            grouped = defaultdict(list)
            for row in rows:
                if row["split_group"] == split_group and row["experiment_type"] == experiment_type:
                    grouped[row["margin_ms"]].append(row["cer"])

            x_values = sorted(grouped)
            x_ticks = x_values
            y_values = [((mean_value(grouped[x]) or 0.0) * 100.0) for x in x_values]
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=experiment_type.capitalize(), color=color)

        axis.axvline(0, color="#d1495b", linewidth=1.2, linestyle="--", alpha=0.8)
        axis.set_title(panel_titles[split_group], fontsize=13, fontweight="bold")
        axis.set_xlabel("Margin (ms)", fontsize=12, fontweight="bold")
        axis.set_ylabel("CER (%)", fontsize=12, fontweight="bold")
        axis.set_xticks(format_margin_ticks(x_ticks))
        axis.tick_params(axis="x", labelsize=11)
        axis.tick_params(axis="y", labelsize=11, labelleft=True)
        axis.grid(alpha=0.25)

    axes[1].legend(fontsize=11)
    fig.suptitle("Effect of Leading and Trailing Margins on CER", fontsize=16, fontweight="bold", y=0.97)
    save_figure(fig, output_dir, "leading_trailing_cer")


def render_first_and_last_word_errors(rows, output_dir):
    metrics = [
        ("first_word_error_rate", "First-word", "#2a9d8f"),
        ("last_word_error_rate", "Last-word", "#7c4dff"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    fig.subplots_adjust(top=0.82, wspace=0.25)
    row_max = 0.0

    for axis, (experiment_type, experiment_label) in zip(axes, [("leading", "Leading"), ("trailing", "Trailing")]):
        x_ticks = []
        for metric_name, label, color in metrics:
            grouped = defaultdict(list)
            for row in rows:
                if row["experiment_type"] == experiment_type:
                    grouped[row["margin_ms"]].append(row[metric_name])

            x_values = sorted(grouped)
            x_ticks = x_values
            y_values = [((mean_value(grouped[x]) or 0.0) * 100.0) for x in x_values]
            if y_values:
                row_max = max(row_max, max(y_values))
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=label, color=color)

        axis.set_title(experiment_label, fontsize=13, fontweight="bold")
        axis.set_xlabel("Margin (ms)", fontsize=12, fontweight="bold")
        axis.set_ylabel("Error Rate (%)", fontsize=12, fontweight="bold")
        axis.set_xticks(format_margin_ticks(x_ticks))
        axis.tick_params(axis="x", labelsize=11)
        axis.tick_params(axis="y", labelsize=11, labelleft=True)
        axis.axvline(0, color="#d1495b", linewidth=1.2, linestyle="--", alpha=0.8)
        axis.grid(alpha=0.25)

    upper = row_max * 1.08 if row_max > 0 else 1.0
    for axis in axes:
        axis.set_ylim(0.0, upper)

    axes[1].legend(fontsize=11)
    fig.suptitle(
        "First- and Last-Word Error under Leading and Trailing Margin Conditions",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    save_figure(fig, output_dir, "first_and_last_word_errors")


def main():
    args = parse_args()
    config = load_config(args.config)
    setup_logging(args.logging_config)
    LOGGER.info("START render_step1_views | dataset=%s exp_subdir=%s output_dir=%s", args.dataset, args.exp_subdir, args.output_dir)

    exp_root = Path(config["exp_path"])
    if args.exp_subdir:
        exp_root = exp_root / args.exp_subdir
    output_dir = Path(args.output_dir) if args.output_dir else exp_root / "analysis"
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")

    raw_rows = load_per_utterance_rows(exp_root, args.dataset)
    rows = collect_margin_rows(raw_rows)

    save_table(tables_dir, "leading_trailing_cer", build_leading_trailing_cer_table(rows))
    save_table(tables_dir, "first_and_last_word_errors", build_first_and_last_word_errors_table(rows))

    render_leading_trailing_cer(rows, figures_dir)
    render_first_and_last_word_errors(rows, figures_dir)
    LOGGER.info("END render_step1_views | dataset=%s exp_subdir=%s output_dir=%s", args.dataset, args.exp_subdir, output_dir)


if __name__ == "__main__":
    main()
