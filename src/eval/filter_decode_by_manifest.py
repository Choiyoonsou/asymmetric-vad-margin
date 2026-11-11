import argparse

from src.common import read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Filter decode JSONL rows to source utterances listed in a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    allowed_ids = {row["utterance_id"] for row in manifest_rows}
    decode_rows = read_jsonl(args.input)
    filtered_rows = [row for row in decode_rows if row["source_utterance_id"] in allowed_ids]

    write_jsonl(args.output, filtered_rows)


if __name__ == "__main__":
    main()
