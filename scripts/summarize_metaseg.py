"""Score exp/metaseg decodes: corpus CER/WER per set x model, and batch-1 latency on the shared timing sample.

One text normalization for every corpus and model: lower-case, drop <unk>/{noise}-style
tags, hyphens to spaces, keep only letters/digits/apostrophes, and re-attach TED-LIUM's
split contractions ("it 's" -> "it's").

    python scripts/summarize_metaseg.py --out exp/metaseg/summary.json
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import editdistance
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def norm(text):
    t = (text or "").lower()
    t = re.sub(r"<[^>]*>|\{[^}]*\}|\[[^\]]*\]", " ", t)
    t = t.replace("-", " ")
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    t = re.sub(r"\s+'", "'", t)
    return re.sub(r"\s+", " ", t).strip()


def boot_ci(e, n, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(B, len(e)))
    r = 100 * e[idx].sum(1) / n[idx].sum(1)
    return [float(x) for x in np.percentile(r, [2.5, 97.5])]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(PROJECT_ROOT / "exp" / "metaseg" / "summary.json"))
    args = p.parse_args()
    groups = defaultdict(list)
    for f in sorted(glob.glob(str(PROJECT_ROOT / "exp" / "metaseg" / "decodes" / "*.jsonl"))):
        for line in open(f):
            x = json.loads(line)
            key = (x["set"], x["model"] + (f"-la{x['lookahead_ms']}" if "lookahead_ms" in x else ""))
            groups[key].append(x)
    out = {}
    for (s, m), rows in sorted(groups.items()):
        ce, cn, we, wn = [], [], [], []
        for x in rows:
            r, h = norm(x["reference"]), norm(x["hypothesis"])
            ce.append(editdistance.eval(r, h)); cn.append(len(r))
            rw, hw = r.split(), h.split()
            we.append(editdistance.eval(rw, hw)); wn.append(len(rw))
        ce, cn, we, wn = map(np.array, (ce, cn, we, wn))
        keep = cn > 0
        t = [x for x in rows if x.get("timing_sample") and x.get("proc_ms") is not None]
        pm = np.array([x["proc_ms"] for x in t]); dur = np.array([x["duration_sec"] for x in t])
        d_all = np.array([x["duration_sec"] for x in rows])
        out[f"{s}|{m}"] = {
            "set": s, "model": m, "n": len(rows), "hours": float(d_all.sum() / 3600),
            "dur_mean": float(d_all.mean()), "dur_p90": float(np.percentile(d_all, 90)),
            "cer": float(100 * ce[keep].sum() / cn[keep].sum()), "cer_ci": boot_ci(ce[keep], cn[keep]),
            "wer": float(100 * we[keep].sum() / wn[keep].sum()), "wer_ci": boot_ci(we[keep], wn[keep]),
            "n_timed": len(t),
            "lat_mean": float(pm.mean()) if len(t) else None, "lat_med": float(np.median(pm)) if len(t) else None,
            "lat_p90": float(np.percentile(pm, 90)) if len(t) else None,
            "lat_max": float(pm.max()) if len(t) else None,
            "rtf": float(pm.sum() / 1000 / dur.sum()) if len(t) else None,
            "timed_dur_mean": float(dur.mean()) if len(t) else None,
        }
        o = out[f"{s}|{m}"]
        print(f"{s:14s} {m:26s} n={o['n']:5d} {o['hours']:.2f}h CER {o['cer']:6.2f} WER {o['wer']:6.2f} | "
              + (f"lat mean {o['lat_mean']:7.1f} med {o['lat_med']:7.1f} p90 {o['lat_p90']:7.1f} ms RTF {o['rtf']:.4f}" if o["lat_mean"] else "no timing"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
