"""Score one system: predictions (+ latency records) -> results/<system_id>.json.

Inputs
  --manifest     data/eval/manifest.jsonl   (id, text, condition, source, speech_start/end ...)
  --predictions  jsonl {"id", "text"} per utterance                      (Track A)
  --latency      jsonl of stream_sim.UttRecord dicts per utterance       (Track B, optional)
  --config       baseline/configs/Bx.json   (system_id, model, mode)

Output (same structure the challenge page embeds as RESULTS):
{
  "system_id", "model", "mode", "hardware", "rtfx",
  "cond": {"clean"|"snr20"|"snr10"|"snr5"|"snr0":
           {"cer", "p50", "p90", "endpoint", "early_cut", "revision", "first_partial", "n"}},
  "noisy_avg": {"cer", "p90"}, "meets_latency_constraint", "by_source": {...}, ...
}
Units: cer / early_cut / revision in %, p50 / p90 / endpoint / first_partial in seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.latency import summarize  # noqa: E402
from evaluation.metrics import CERAccumulator, utterance_cer  # noqa: E402
from evaluation.normalizer import NORMALIZER_VERSION  # noqa: E402
from evaluation.stream_sim import load_config  # noqa: E402

CONDITIONS = ["clean", "snr20", "snr10", "snr5", "snr0"]
NOISY = ["snr20", "snr10", "snr5", "snr0"]


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def score(manifest, preds, records, mode, limit):
    streaming = mode == "streaming"
    cer_by = defaultdict(CERAccumulator)
    cer_src = defaultdict(CERAccumulator)
    rec_by = defaultdict(list)
    missing, worst = [], []
    for utt in manifest:
        cond, uid = utt["condition"], utt["id"]
        hyp = preds.get(uid)
        if hyp is None:
            missing.append(uid)
            hyp = ""
        cer_by[cond].add(utt["text"], hyp)
        cer_src[(cond, utt.get("source", "?"))].add(utt["text"], hyp)
        worst.append((utterance_cer(utt["text"], hyp), uid, cond, utt["text"], hyp))
        if uid in records:
            rec_by[cond].append(records[uid])

    cond_out = {}
    for c in CONDITIONS:
        if c not in cer_by:
            continue
        row = {"cer": round(cer_by[c].cer, 2), "p50": None, "p90": None, "endpoint": None,
               "early_cut": None, "revision": None, "first_partial": None, "premature_partial": None,
               "n": cer_by[c].n_utts}
        if rec_by.get(c):
            s = summarize(rec_by[c], streaming)
            row.update({k: s[k] for k in ("p50", "p90", "endpoint", "early_cut", "revision",
                                          "first_partial", "premature_partial")})
            row["rtfx"] = s["rtfx"]
            row["n_latency"] = s["n"]
        cond_out[c] = row

    noisy = [c for c in NOISY if c in cond_out]
    noisy_avg = None
    meets = None
    if noisy:
        noisy_avg = {"cer": round(sum(cond_out[c]["cer"] for c in noisy) / len(noisy), 2)}
        p90s = [cond_out[c]["p90"] for c in noisy]
        if all(p is not None for p in p90s):
            noisy_avg["p90"] = round(sum(p90s) / len(p90s), 4)
            noisy_avg["p90_max"] = round(max(p90s), 4)
            meets = all(p <= limit for p in p90s)
        else:
            noisy_avg["p90"] = None

    all_recs = [r for c in rec_by for r in rec_by[c]]
    audio = sum(r["audio_duration"] for r in all_recs)
    proc = sum(r["processing_time"] for r in all_recs)
    rtfx = round(audio / proc, 2) if proc > 0 else None

    by_source = defaultdict(dict)
    for (c, src), acc in cer_src.items():
        by_source[src][c] = round(acc.cer, 2)

    worst.sort(reverse=True)
    examples = [{"id": u, "condition": c, "cer": round(e, 1), "ref": ref, "hyp": hyp}
                for e, u, c, ref, hyp in worst[:20]]
    return cond_out, noisy_avg, meets, rtfx, dict(by_source), missing, examples


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--latency", help="latency jsonl (Track B)")
    ap.add_argument("--config", required=True, help="baseline/configs/Bx.json")
    ap.add_argument("--hardware", default="NVIDIA L4")
    ap.add_argument("--env", help="json with runtime versions to embed")
    ap.add_argument("--out", help="default results/<system_id>.json")
    ap.add_argument("--latency-config", default=None)
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    lcfg = load_config(args.latency_config) if args.latency_config else load_config()
    limit = lcfg["latency_constraint_p90_sec"]

    manifest = list(read_jsonl(args.manifest))
    preds = {p["id"]: p["text"] for p in read_jsonl(args.predictions)}
    records = {r["id"]: r for r in read_jsonl(args.latency)} if args.latency else {}
    cond, noisy_avg, meets, rtfx, by_source, missing, examples = score(manifest, preds, records, cfg["mode"], limit)

    result = {
        "system_id": cfg["system_id"], "model": cfg["model"], "mode": cfg["mode"],
        "hardware": args.hardware, "rtfx": rtfx, "cond": cond,
        "noisy_avg": noisy_avg, "latency_constraint_p90_sec": limit,
        "meets_latency_constraint": meets,
        "by_source": by_source, "normalizer_version": NORMALIZER_VERSION,
        "missing_predictions": missing, "worst_examples": examples,
    }
    if args.env:
        with open(args.env, encoding="utf-8") as f:
            result["environment"] = json.load(f)

    out = args.out or os.path.join("results", f"{cfg['system_id']}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[{cfg['system_id']}] {cfg['model']} ({cfg['mode']}) -> {out}")
    print(f"  {'cond':7s} {'CER%':>6s} {'p50':>6s} {'p90':>6s} {'endpt':>6s} {'early%':>6s} {'rev%':>6s} {'first':>6s} {'prem%':>6s}")
    for c, r in cond.items():
        f = lambda v, nd=2: "-" if v is None else f"{v:.{nd}f}"
        print(f"  {c:7s} {f(r['cer']):>6s} {f(r['p50'],3):>6s} {f(r['p90'],3):>6s} {f(r['endpoint'],3):>6s} "
              f"{f(r['early_cut'],1):>6s} {f(r['revision'],1):>6s} {f(r['first_partial'],3):>6s} {f(r['premature_partial'],1):>6s}")
    if noisy_avg:
        print(f"  noisy avg CER {noisy_avg['cer']}%  avg p90 {noisy_avg.get('p90')}  "
              f"constraint {'OK' if meets else ('FAIL' if meets is False else 'n/a')}  RTFx {rtfx}")
    if missing:
        print(f"  WARNING: {len(missing)} utterances had no prediction (scored as empty)")


if __name__ == "__main__":
    main()
