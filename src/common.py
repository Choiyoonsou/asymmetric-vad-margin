import json
import logging.config
import os
from pathlib import Path

import yaml


PATH_KEYS = {
    "artifacts_path",
    "exp_path",
}


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ValueError("YAML file is empty: %s" % path)
    return data


def ensure_dir(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(value, base_dir):
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return str(candidate)


def load_config(path):
    config_path = Path(path).resolve()
    config = load_yaml(config_path)
    project_root = config_path.parent.parent.resolve()

    for key in PATH_KEYS:
        config[key] = resolve_path(config[key], project_root)

    datasets = config["datasets"]
    for dataset_name in datasets:
        dataset = datasets[dataset_name]
        dataset["data_path"] = resolve_path(dataset["data_path"], project_root)
        dataset["textgrid_path"] = resolve_path(dataset["textgrid_path"], project_root)

    for section_name in ("manifest", "oracle"):
        section = config[section_name]
        for key, value in section.items():
            if key.endswith("_path") and isinstance(value, str):
                section[key] = resolve_path(value, project_root)

    return config


def get_dataset_config(config, dataset_name=None):
    datasets = config["datasets"]

    if dataset_name is not None:
        if dataset_name not in datasets:
            raise ValueError("Unknown dataset: %s" % dataset_name)
        return dataset_name, datasets[dataset_name]

    if len(datasets) == 1:
        only_name = next(iter(datasets))
        return only_name, datasets[only_name]

    raise ValueError("Multiple datasets exist. Please set --dataset.")


def setup_logging(logging_config_path):
    config_path = Path(logging_config_path)
    if not config_path.exists():
        raise FileNotFoundError("Logging config not found: %s" % config_path)

    config = load_yaml(config_path)
    log_path = resolve_path(config["log_path"], config_path.parent.parent.resolve())
    config["log_path"] = log_path
    ensure_dir(log_path)

    if "file" in config["handlers"]:
        config["handlers"]["file"]["filename"] = str(Path(log_path) / "vad_experiments.log")

    logging.config.dictConfig(config)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_cuda_visible_devices(config):
    if config["asr"]["device"] != "cuda":
        return

    device_index = config["asr"]["device_index"]
    if isinstance(device_index, list):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in device_index)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)


def get_split_group(split_name):
    # Groups any split whose name contains "clean" (e.g. LibriSpeech's
    # dev-clean/test-clean or KsponSpeech's eval_clean) under "clean";
    # everything else (dev-other, test-other, eval_other, ...) under "other".
    # vad_policy in conf/base.yaml is keyed on these two groups.
    if "clean" in split_name:
        return "clean"
    return "other"
