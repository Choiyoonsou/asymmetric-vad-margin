import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.common import get_dataset_config, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Read pipeline values from conf/base.yaml.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--dataset", default=None, help="Dataset name under datasets.")
    parser.add_argument("--model", default=None, help="ASR backbone override, so asr_tag follows the run's actual backbone.")
    parser.add_argument(
        "--field",
        required=True,
        choices=["splits", "asr_tag", "vad_policy_name", "artifacts_path", "exp_path", "manifest_output_path"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.field == "splits":
        _, dataset = get_dataset_config(config, args.dataset)
        for split in dataset["splits"]:
            print(split)
        return

    if args.field == "asr_tag":
        model_name = args.model or config["asr"]["model_name"]
        print(model_name.replace("-", "_"))
        return

    if args.field == "vad_policy_name":
        print(config["vad_policy"]["policy_name"])
        return

    if args.field == "artifacts_path":
        print(config["artifacts_path"])
        return

    if args.field == "exp_path":
        print(config["exp_path"])
        return

    if args.field == "manifest_output_path":
        print(config["manifest"]["output_path"])
        return


if __name__ == "__main__":
    main()
