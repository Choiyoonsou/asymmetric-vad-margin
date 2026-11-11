import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (
    ensure_dir,
    get_dataset_config,
    load_config,
    setup_logging,
    write_jsonl,
)
from src.data import ami, ksponspeech, librispeech, tedlium


LOGGER = logging.getLogger("build_manifest")

MANIFEST_BUILDERS = {
    "librispeech": librispeech.build_manifest_entries,
    # Same on-disk layout as librispeech (.flac + <id> <transcript> .trans.txt).
    "zeroth_korean": librispeech.build_manifest_entries,
    "ksponspeech": ksponspeech.build_manifest_entries,
    # Reads the clips + TextGrids written by scripts/data/prepare_ami.py.
    "ami": ami.build_manifest_entries,
    # Reads the clips + TextGrids written by scripts/data/prepare_tedlium.py.
    "tedlium": tedlium.build_manifest_entries,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build per-utterance JSONL manifests for a registered dataset.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--logging-config", default=str(PROJECT_ROOT / "conf" / "logging.yaml"))
    parser.add_argument("--dataset", default=None, help="Dataset name under conf/base.yaml datasets.")
    parser.add_argument("--splits", nargs="+", help="Split names to build.")
    parser.add_argument("--limit", type=int, default=None, help="Optional utterance limit per split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(args.logging_config)
    LOGGER.info("START build_manifest | dataset=%s splits=%s limit=%s", args.dataset, args.splits, args.limit)

    dataset_name, dataset = get_dataset_config(config, args.dataset)
    splits = args.splits or dataset["splits"]
    manifest_dir = ensure_dir(config["manifest"]["output_path"])

    if dataset_name not in MANIFEST_BUILDERS:
        raise ValueError(
            "No manifest builder registered for dataset=%s. Supported: %s"
            % (dataset_name, ", ".join(MANIFEST_BUILDERS))
        )
    build_manifest_entries = MANIFEST_BUILDERS[dataset_name]

    for split in splits:
        limit = args.limit
        LOGGER.info("Building manifest for dataset=%s split=%s limit=%s", dataset_name, split, limit)
        entries, stats = build_manifest_entries(
            data_path=dataset["data_path"],
            textgrid_path=dataset["textgrid_path"],
            split=split,
            skip_missing_textgrid=config["manifest"]["skip_missing_textgrid"],
            limit=limit,
        )
        output_path = manifest_dir / f"{split}.jsonl"
        write_jsonl(output_path, [entry.to_dict() for entry in entries])
        LOGGER.info(
            "Manifest written to %s | audio_found=%d valid=%d missing_textgrid=%d missing_transcript=%d invalid_audio=%d",
            output_path,
            stats["audio_found"],
            stats["valid"],
            stats["missing_textgrid"],
            stats["missing_transcript"],
            stats["invalid_audio"],
        )

        if stats["valid"] == 0:
            raise ValueError(
                "No valid manifest entries for dataset=%s split=%s. "
                "Check audio/textgrid coverage. missing_textgrid=%d invalid_audio=%d"
                % (dataset_name, split, stats["missing_textgrid"], stats["invalid_audio"])
            )

    LOGGER.info("END build_manifest | dataset=%s splits=%s", dataset_name, splits)


if __name__ == "__main__":
    main()
