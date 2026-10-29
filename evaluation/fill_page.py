"""Inject measured results into the challenge page (RESULTS / EXAMPLES / CONFIG.env blocks).

    python evaluation/fill_page.py --page index.html --results results \
        [--jitter results/jitter_B2.json] [--env runs/B1/environment.json] [--n-examples 8] \
        [--also submission/Team_이다혜_202422051.html]

The page keeps its data in three JS literals; this script rewrites them in place:
  const RESULTS = [...];   from results/B*.json (cond values copied verbatim)
  const EXAMPLES = [...];  worst-CER utterances per noisy condition, same utterance's clean output as reference
  CONFIG.env [...]         GPU / CUDA / torch / transformers / vLLM / qwen-asr versions from environment.json
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NAMES = {"offline": "오프라인", "streaming": "스트리밍"}
COND_LABEL = {"clean": "Clean", "snr20": "SNR 20 dB", "snr10": "SNR 10 dB", "snr5": "SNR 5 dB", "snr0": "SNR 0 dB"}
KEYS = ("cer", "p50", "p90", "start_err", "endpoint", "missed", "early_cut", "revision", "first_partial", "premature_partial")


def js(obj):
    return json.dumps(obj, ensure_ascii=False)


def results_block(results):
    rows = []
    for r in results:
        name = r["model"].split("/")[-1] + " " + NAMES.get(r["mode"], r["mode"])
        cond = {c: {k: r["cond"][c].get(k) for k in KEYS} for c in ("clean", "snr20", "snr10", "snr5", "snr0") if c in r["cond"]}
        rows.append(f'  {{ id:{js(r["system_id"])}, name:{js(name)}, rtfx:{js(r.get("rtfx"))}, cond:{js(cond)} }}')
    return "const RESULTS = [\n" + ",\n".join(rows) + "\n];"


def examples_block(results, n):
    out = []
    for r in results:
        clean_out = {}
        worst = r.get("worst_examples", [])
        for e in worst:
            if e["condition"] == "clean":
                clean_out[e["id"].rsplit("_", 1)[0]] = e["hyp"]
        picked = [e for e in worst if e["condition"] != "clean"][: max(1, n // max(1, len(results)))]
        for e in picked:
            c = r["cond"].get(e["condition"], {})
            note = f'발화 CER {e["cer"]}%'
            if c.get("endpoint") is not None:
                note += f', 조건 종료 검출 오차 중앙값 {c["endpoint"]:.2f}s'
            out.append({"condition": COND_LABEL.get(e["condition"], e["condition"]), "ref": e["ref"], "hyp": e["hyp"],
                        "system": r["system_id"], "note": note})
    return "const EXAMPLES = [\n" + ",\n".join("  " + js(e) for e in out) + "\n];"


def env_rows(env):
    def g(k):
        return env.get(k)
    gpu = g("gpu") or "NVIDIA L4 (Colab)"
    return [["GPU", gpu], ["Python", g("python")], ["CUDA", g("cuda")], ["torch", g("torch")],
            ["transformers", g("transformers")], ["vLLM", g("vllm")], ["qwen-asr", g("qwen-asr")]]


def replace_block(html, name, new):
    pat = re.compile(rf"const {name} = \[.*?\n\];", re.S)
    if not pat.search(html):
        sys.exit(f"block {name} not found")
    return pat.sub(lambda m: new, html, count=1)


def replace_env(html, rows, keep_tail=("VAD", "측정")):
    pat = re.compile(r"(  env: \[\n)(.*?)(\n  \]\n)", re.S)
    m = pat.search(html)
    if not m:
        sys.exit("CONFIG.env not found")
    old = [l for l in m.group(2).split("\n") if l.strip()]
    tail = [l.rstrip(",") for l in old if any(f'["{k}"' in l for k in keep_tail)]
    lines = [f"    {js(r)}" for r in rows] + [f"    {l.strip()}" for l in tail]
    return html[: m.start(2)] + ",\n".join(lines) + html[m.end(2):]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", default="index.html")
    ap.add_argument("--results", default="results")
    ap.add_argument("--jitter", default=None)
    ap.add_argument("--env", default=None, help="environment.json from any run")
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--examples", default=None, help="curated examples JSON (from evaluation/pick_examples.py); overrides worst-CER picks")
    ap.add_argument("--also", nargs="*", default=[], help="extra copies to write (e.g. the submission file)")
    a = ap.parse_args()

    results = []
    for p in sorted(glob.glob(os.path.join(a.results, "B*.json"))):
        with open(p, encoding="utf-8") as f:
            results.append(json.load(f))
    if not results:
        sys.exit("no results/B*.json")
    html = open(a.page, encoding="utf-8").read()
    html = replace_block(html, "RESULTS", results_block(results))
    if a.examples:
        ex = json.load(open(a.examples, encoding="utf-8"))
        html = replace_block(html, "EXAMPLES", "const EXAMPLES = [\n" + ",\n".join("  " + js(e) for e in ex) + "\n];")
    else:
        html = replace_block(html, "EXAMPLES", examples_block(results, a.n_examples))
    env = None
    if a.env:
        env = json.load(open(a.env, encoding="utf-8"))
    elif results[0].get("environment"):
        env = results[0]["environment"]
    if env:
        html = replace_env(html, env_rows(env))
    if a.jitter:
        j = json.load(open(a.jitter, encoding="utf-8"))
        note = (f'["측정", "가상 시계 시뮬레이션, 발화당 1회 측정. 흔들림 검증: B2 {j["n"]}개 평가 파일({j["n"]//5}발화 × 5조건) 3회 반복 중앙값 대비 '
                f'최종 지연 차이 중앙값 {j["median_abs_diff_s"]*1000:.0f} ms, p90 {j["p90_single_s"]:.3f} s vs {j["p90_median3_s"]:.3f} s"]')
        html = re.sub(r'\["측정", "[^"]*"\]', note, html, count=1)
    for dst in [a.page] + a.also:
        with open(dst, "w", encoding="utf-8") as f:
            f.write(html)
        print("wrote", dst)


if __name__ == "__main__":
    main()
