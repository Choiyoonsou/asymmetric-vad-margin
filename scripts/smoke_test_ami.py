"""Validates the AMI ingestion approach end-to-end on real audio, before
committing to building the full src/data/ami.py manifest pipeline.

For one meeting, one speaker channel: parses the NXT XML word + segment
annotations, slices a handful of segments (with their natural padding) out of
both the near-mic (Headset) and far-mic (Array1-01 / SDM) recordings, and
decodes each clip with the given backbones to sanity-check that (a) our
segment-extraction logic lines up with real speech and (b) CER is directionally
sane (IHM should beat SDM at baseline -- if it doesn't, the Headset<->speaker
channel mapping assumed here is probably wrong).
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.asr import BACKBONE_REGISTRY, RUNNER_CLASSES, AsrSettings
from src.audio.io import read_audio, slice_audio, write_audio
from src.metrics import compute_text_metrics

NS = {"nite": "http://nite.sourceforge.net/"}
CHILD_RANGE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?$")

# AMI scenario-meeting convention: participant roles A/B/C/D map to
# Headset-0..3 in recording order. Not documented as a hard guarantee --
# this smoke test's own IHM-vs-SDM CER gap is the check that it's right.
SPEAKER_TO_HEADSET = {"A": 0, "B": 1, "C": 2, "D": 3}


def parse_words(words_xml_path):
    tree = ET.parse(words_xml_path)
    root = tree.getroot()
    words = []  # list of (nite_id, tag, text, start, end, is_punc)
    for el in root:
        tag = el.tag
        nite_id = el.attrib.get("{http://nite.sourceforge.net/}id")
        if tag == "w":
            start = el.attrib.get("starttime")
            end = el.attrib.get("endtime")
            words.append((nite_id, "w", (el.text or "").strip(), start, end, el.attrib.get("punc") == "true"))
        else:
            # vocalsound, disfmarker, pause, etc. -- keep as placeholder so
            # segment word-ranges still resolve, but they contribute no text.
            words.append((nite_id, tag, "", None, None, False))
    return words


def parse_segments(segments_xml_path):
    tree = ET.parse(segments_xml_path)
    root = tree.getroot()
    segments = []
    for seg in root.findall("segment"):
        href = seg.find("nite:child", NS).attrib["href"]
        m = CHILD_RANGE.search(href)
        start_id, end_id = m.group(1), m.group(2) or m.group(1)
        segments.append(
            {
                "start_id": start_id,
                "end_id": end_id,
                "transcriber_start": float(seg.attrib["transcriber_start"]),
                "transcriber_end": float(seg.attrib["transcriber_end"]),
            }
        )
    return segments


def build_transcript(words, start_id, end_id):
    id_index = {w[0]: i for i, w in enumerate(words)}
    if start_id not in id_index or end_id not in id_index:
        return None
    i0, i1 = id_index[start_id], id_index[end_id]
    tokens = []
    for _, tag, text, _, _, is_punc in words[i0 : i1 + 1]:
        if tag != "w" or not text:
            continue
        if is_punc and tokens:
            tokens[-1] = tokens[-1] + text
        else:
            tokens.append(text)
    return " ".join(tokens)


def select_pilot_segments(words, segments, min_words=4, min_dur=1.5, max_dur=8.0, limit=15, pad_sec=0.0):
    picked = []
    for seg in segments:
        dur = seg["transcriber_end"] - seg["transcriber_start"]
        if not (min_dur <= dur <= max_dur):
            continue
        text = build_transcript(words, seg["start_id"], seg["end_id"])
        if not text or len(text.split()) < min_words:
            continue
        picked.append({**seg, "text": text})
        if len(picked) >= limit:
            break
    return picked


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting", default="ES2004a")
    parser.add_argument("--speaker", default="A", choices=list(SPEAKER_TO_HEADSET))
    parser.add_argument("--ami-root", default=str(PROJECT_ROOT / "data" / "AMI_raw"))
    parser.add_argument("--backbones", nargs="+", default=["wav2vec2-base-960h", "whisper-base"])
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    ami_root = Path(args.ami_root)
    ann_root = ami_root / "annotations" / "unzipped"
    audio_root = ami_root / "audio" / args.meeting

    words = parse_words(ann_root / "words" / f"{args.meeting}.{args.speaker}.words.xml")
    segments = parse_segments(ann_root / "segments" / f"{args.meeting}.{args.speaker}.segments.xml")
    pilot = select_pilot_segments(words, segments, limit=args.limit)
    print(f"[selected] {len(pilot)} segments from {args.meeting}/{args.speaker}")
    if not pilot:
        print("No usable segments found -- check annotation paths.")
        return

    headset_idx = SPEAKER_TO_HEADSET[args.speaker]
    conditions = {
        "IHM": audio_root / f"{args.meeting}.Headset-{headset_idx}.wav",
        "SDM": audio_root / f"{args.meeting}.Array1-01.wav",
    }
    for label, path in conditions.items():
        if not path.exists():
            print(f"[skip condition] {label}: missing {path}")

    clip_dir = PROJECT_ROOT / "artifacts" / "ami_smoke_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    audio_cache = {}
    for label, path in conditions.items():
        if path.exists():
            audio_cache[label] = read_audio(path)

    for model_name in args.backbones:
        if model_name not in BACKBONE_REGISTRY:
            print(f"[skip] unknown backbone: {model_name}")
            continue
        backbone = BACKBONE_REGISTRY[model_name]
        runner_cls = RUNNER_CLASSES[backbone["family"]]
        settings = AsrSettings(
            model_name=model_name, hf_id=backbone["hf_id"], device=args.device, device_index=args.device_index
        )
        print(f"\n=== {model_name} ({backbone['family']}) ===")
        try:
            runner = runner_cls(settings)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL to load: {type(exc).__name__}: {exc}")
            continue

        for label in conditions:
            if label not in audio_cache:
                continue
            audio, sample_rate = audio_cache[label]
            cers = []
            for i, seg in enumerate(pilot):
                clip = slice_audio(audio, sample_rate, seg["transcriber_start"], seg["transcriber_end"])
                clip_path = clip_dir / f"{args.meeting}_{args.speaker}_{label}_{i}.wav"
                write_audio(clip_path, clip, sample_rate)
                try:
                    result = runner.transcribe(str(clip_path))
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{label}] seg{i} FAIL {type(exc).__name__}: {exc}")
                    continue
                metrics = compute_text_metrics(seg["text"], result["hypothesis"])
                cers.append(metrics["cer"])
                if i < 3:
                    print(f"  [{label}] ref={seg['text']!r}")
                    print(f"  [{label}] hyp={result['hypothesis']!r}  cer={metrics['cer']:.3f}")
            if cers:
                print(f"  [{label}] mean CER over {len(cers)} segments: {sum(cers) / len(cers):.3f}")


if __name__ == "__main__":
    main()
