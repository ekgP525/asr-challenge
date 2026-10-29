"""Pick illustrative examples for the challenge page from the baseline runs.

    python evaluation/pick_examples.py --system B2 --out results/examples.json

Three kinds, all from the same system (default B2, 1.7B streaming):
  1. degradation: utterances that are NOT cut early in any condition, correct at clean
     and increasingly wrong towards 0 dB (clean / 10 dB / 0 dB outputs shown);
  2. early cut: the endpointer fired before the true end, so the tail is missing;
  3. premature partial: the first partial appeared before the true speech start
     (text produced from leading noise), with the final output for comparison.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import utterance_cer  # noqa: E402

LABEL = {"clean": "Clean", "snr20": "SNR 20 dB", "snr10": "SNR 10 dB", "snr5": "SNR 5 dB", "snr0": "SNR 0 dB"}


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", default="B2")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--manifest", default="data/eval/manifest.jsonl")
    ap.add_argument("--n-degrade", type=int, default=2)
    ap.add_argument("--out", default="results/examples.json")
    a = ap.parse_args()
    man = {u["id"]: u for u in read_jsonl(a.manifest)}
    preds = {p["id"]: p["text"] for p in read_jsonl(os.path.join(a.runs, a.system, "predictions.jsonl"))}
    recs = {r["id"]: r for r in read_jsonl(os.path.join(a.runs, a.system, "latency.jsonl"))}
    base_ids = sorted({i.rsplit("_", 1)[0] for i in man})

    def cut(i):
        r = recs.get(i)
        if not r:
            return True
        vend = r["vad_end"] if r["vad_end"] is not None else r["audio_duration"]
        return vend < man[i]["speech_end"]

    out = []
    # 1. degradation
    cands = []
    for b in base_ids:
        ids = [f"{b}_{c}" for c in ("clean", "snr10", "snr0")]
        if any(i not in preds or cut(i) for i in ids):
            continue
        cers = [utterance_cer(man[i]["text"], preds[i]) for i in ids]
        if cers[0] <= 2 and cers[1] < cers[2] and 15 <= cers[2] <= 60 and len(man[ids[0]]["text"]) <= 60:
            cands.append((cers[2], b, cers))
    cands.sort(reverse=True)
    for _, b, cers in cands[: a.n_degrade]:
        for c, cer in zip(("clean", "snr10", "snr0"), cers):
            i = f"{b}_{c}"
            noise = man[i]["noise"]["type"] if c != "clean" else "noise floor"
            out.append({"condition": LABEL[c], "ref": man[i]["text"], "hyp": preds[i], "system": a.system,
                        "note": f"같은 발화, 잡음 {noise}. 발화 CER {cer:.1f}%"})
    # 2. early cut (clean, so the cut is the pause itself, not noise)
    cuts = []
    for b in base_ids:
        i = f"{b}_clean"
        if i in preds and cut(i) and len(man[i]["text"]) <= 70:
            r = recs[i]
            cuts.append((man[i]["speech_end"] - r["vad_end"], i))
    cuts.sort()
    for gap, i in cuts[len(cuts) // 2: len(cuts) // 2 + 1]:   # a typical (median-gap) cut, not the extreme one
        r = recs[i]
        out.append({"condition": LABEL["clean"], "ref": man[i]["text"], "hyp": preds[i], "system": a.system,
                    "note": f"조기 절단: 문장 내 쉼에서 VAD가 정답 종료보다 {gap:.2f}초 먼저 끊어 뒷부분이 빠짐"})
    # 3. premature partial at 0 dB
    prem = []
    for b in base_ids:
        i = f"{b}_snr0"
        r = recs.get(i)
        if (r and r.get("t_first_partial") is not None and r["t_first_partial"] < man[i]["speech_start"]
                and len(man[i]["text"]) <= 60 and utterance_cer(man[i]["text"], preds[i]) < 30):   # final still mostly right
            first = next(p for p in r["partials"] if p["text"].strip())
            prem.append((man[i]["speech_start"] - r["t_first_partial"], i, first["text"]))
    prem.sort(reverse=True)
    for lead, i, first_text in prem[:1]:
        out.append({"condition": LABEL["snr0"], "ref": man[i]["text"], "hyp": f"[첫 부분결과] {first_text} → [최종] {preds[i]}",
                    "system": a.system, "note": f"발화 시작 {lead:.2f}초 전에 잡음({man[i]['noise']['type']})만 듣고 첫 부분결과를 냄"})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for e in out:
        print(f"[{e['condition']}] {e['note']}\n  REF {e['ref']}\n  HYP {e['hyp']}")
    print(f"\nwrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
