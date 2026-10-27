"""Score one system: predictions (+ optional latency log) -> results/<system_id>.json.

Inputs
------
--manifest     data/eval/manifest.jsonl  (id, text, terms, condition, ...)
--predictions  jsonl with {"id": ..., "text": ...} per utterance
--latency      jsonl written by baseline runner: UttLatency.to_dict() per utterance
               (Track B is skipped when omitted)
--config       baseline/configs/Bx.json  (system_id, model, mode, context_prompt)

Output follows the result JSON format in the project spec.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.latency import load_config as load_latency_config, summarize  # noqa: E402
from evaluation.metrics import CERAccumulator, TermRecallAccumulator  # noqa: E402
from evaluation.normalizer import NORMALIZER_VERSION  # noqa: E402

CONDITIONS = ["clean", "snr20", "snr10", "snr5", "snr0"]
NOISY = ["snr20", "snr10", "snr5", "snr0"]


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def score_track_a(manifest: list, preds: dict):
    cer_acc = defaultdict(CERAccumulator)
    tr_acc = defaultdict(TermRecallAccumulator)
    missing = []
    for utt in manifest:
        cond = utt["condition"]
        hyp = preds.get(utt["id"])
        if hyp is None:
            missing.append(utt["id"])
            hyp = ""  # missing prediction counts as fully wrong
        cer_acc[cond].add(utt["text"], hyp)
        tr_acc[cond].add(utt["id"], hyp, utt.get("terms", []))

    out = {}
    for cond in CONDITIONS:
        if cond in cer_acc:
            out[cond] = {
                "cer": round(cer_acc[cond].cer, 2),
                "term_recall": round(tr_acc[cond].term_recall, 2),
                "n_utts": cer_acc[cond].n_utts,
            }
    noisy = [c for c in NOISY if c in out]
    if noisy:
        out["noisy_avg"] = {
            "cer": round(sum(out[c]["cer"] for c in noisy) / len(noisy), 2),
            "term_recall": round(sum(out[c]["term_recall"] for c in noisy) / len(noisy), 2),
            "conditions": noisy,
        }
    missed_terms = {c: tr_acc[c].missed for c in tr_acc}
    return out, missing, missed_terms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--latency", help="latency jsonl (Track B)")
    ap.add_argument("--config", required=True, help="baseline/configs/Bx.json")
    ap.add_argument("--hardware", default="NVIDIA L4")
    ap.add_argument("--env", help="json file with runtime versions (torch, vllm, ...) to embed")
    ap.add_argument("--out", help="output path (default results/<system_id>.json)")
    ap.add_argument("--latency-config", default=None)
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    lat_cfg = load_latency_config(args.latency_config) if args.latency_config else load_latency_config()

    manifest = list(read_jsonl(args.manifest))
    preds = {p["id"]: p["text"] for p in read_jsonl(args.predictions)}
    track_a, missing, missed_terms = score_track_a(manifest, preds)

    result = {
        "system_id": cfg["system_id"],
        "model": cfg["model"],
        "mode": cfg["mode"],
        "context_prompt": bool(cfg.get("context_prompt", False)),
        "hardware": args.hardware,
        "normalizer_version": NORMALIZER_VERSION,
        "track_a": track_a,
        "track_b": None,
        "meets_latency_constraint": None,
        "missing_predictions": missing,
        "missed_terms": missed_terms,
    }
    if args.env:
        with open(args.env, encoding="utf-8") as f:
            result["environment"] = json.load(f)

    if args.latency:
        records = list(read_jsonl(args.latency))
        tb = summarize(records)
        result["track_b"] = tb
        limit = lat_cfg["latency_constraint_p90_sec"]
        result["meets_latency_constraint"] = (
            tb["final_latency_p90"] is not None and tb["final_latency_p90"] <= limit
        )

    out = args.out or os.path.join("results", f"{cfg['system_id']}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[{cfg['system_id']}] wrote {out}")
    for cond, m in track_a.items():
        print(f"  {cond:9s} CER {m['cer']:6.2f}%  TermRecall {m['term_recall']:6.2f}%")
    if result["track_b"]:
        tb = result["track_b"]
        print(f"  latency p50/p90 {tb['final_latency_p50']}/{tb['final_latency_p90']} s  "
              f"first-partial {tb['first_partial_latency']}  RTFx {tb['rtfx']}  "
              f"constraint {'OK' if result['meets_latency_constraint'] else 'FAIL'}")
    if missing:
        print(f"  WARNING: {len(missing)} utterances had no prediction (scored as empty)")


if __name__ == "__main__":
    main()
