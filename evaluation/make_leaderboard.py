"""results/*.json -> leaderboard (markdown + json) and the latency/CER trajectory plot.

Ranking rule
------------
* Constraint: p90 final latency <= limit (1.0 s) in *each* of the four noisy
  conditions. A system that misses it in any condition is listed separately.
* Primary: average CER over the four noisy conditions (lower is better).
* Tie-break: average p90 final latency over the noisy conditions.

Plot: one path per system through (p90, CER) for clean -> snr20 -> snr10 -> snr5 -> snr0.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.stream_sim import load_config  # noqa: E402

CONDITIONS = ["clean", "snr20", "snr10", "snr5", "snr0"]
NOISY = CONDITIONS[1:]


def load_results(results_dir: str) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(p) == "leaderboard.json":
            continue
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def rank(results: list):
    def key(r):
        a = r.get("noisy_avg") or {}
        return (a.get("cer", 1e9), a.get("p90") if a.get("p90") is not None else 1e9)
    main = sorted([r for r in results if r.get("meets_latency_constraint") is True], key=key)
    other = sorted([r for r in results if r.get("meets_latency_constraint") is not True], key=key)
    return main, other


def row(r: dict, rank_no):
    c = r["cond"]
    a = r.get("noisy_avg") or {}
    return {
        "rank": rank_no, "system_id": r["system_id"], "model": r["model"], "mode": r["mode"],
        "noisy_cer": a.get("cer"), "noisy_p90": a.get("p90"), "noisy_p90_max": a.get("p90_max"),
        "clean_cer": c.get("clean", {}).get("cer"),
        "cond": {k: c[k] for k in CONDITIONS if k in c},
        "rtfx": r.get("rtfx"), "meets_latency_constraint": r.get("meets_latency_constraint"),
        "hardware": r.get("hardware"),
    }


def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def to_markdown(main_rows, other_rows, limit) -> str:
    hdr = ("| # | System | Model | Mode | Noisy CER (%) ↓ | Clean CER | CER 20/10/5/0 dB | "
           "Noisy p90 avg (s) ↓ | p90 max (s) | Endpoint err 0 dB (s) | Early cut 0 dB (%) | "
           "Revision 0 dB (%) | RTFx ↑ |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    def line(x):
        c = x["cond"]
        cers = "/".join(_f(c.get(k, {}).get("cer"), 1) for k in NOISY)
        z = c.get("snr0", {})
        return (f"| {x['rank'] if x['rank'] is not None else '-'} | {x['system_id']} | {x['model']} | {x['mode']} | "
                f"{_f(x['noisy_cer'])} | {_f(x['clean_cer'])} | {cers} | {_f(x['noisy_p90'], 3)} | "
                f"{_f(x['noisy_p90_max'], 3)} | {_f(z.get('endpoint'), 3)} | {_f(z.get('early_cut'), 1)} | "
                f"{_f(z.get('revision'), 1)} | {_f(x['rtfx'], 1)} |\n")

    md = f"## Main ranking (p90 final latency ≤ {limit:.1f} s in every noisy condition)\n\n" + hdr
    md += "".join(line(x) for x in main_rows) if main_rows else "| - | (no system meets the constraint yet) ||||||||||||\n"
    if other_rows:
        md += "\n## Outside the latency constraint / no Track B measurement\n\n" + hdr
        md += "".join(line(x) for x in other_rows)
    return md


def plot_paths(rows, limit, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    drew = False
    for r in rows:
        pts = [(r["cond"][k]["p90"], r["cond"][k]["cer"], k) for k in CONDITIONS
               if k in r["cond"] and r["cond"][k].get("p90") is not None]
        if not pts:
            continue
        drew = True
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"{r['system_id']} ({r['mode']})")
        for x, y, k in pts:
            ax.annotate(k.replace("snr", "") + ("" if k == "clean" else " dB"), (x, y),
                        textcoords="offset points", xytext=(4, 3), fontsize=7)
    if not drew:
        print("no system with latency per condition; skipping plot", file=sys.stderr)
        return
    ax.axvline(limit, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("p90 final latency (s)")
    ax.set_ylabel("CER (%)")
    ax.set_title("Clean → 0 dB: accuracy and latency move together")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-md", default="results/leaderboard.md")
    ap.add_argument("--out-json", default="results/leaderboard.json")
    ap.add_argument("--out-png", default="results/latency_cer_paths.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    limit = load_config()["latency_constraint_p90_sec"]
    results = load_results(args.results_dir)
    if not results:
        print(f"no result files in {args.results_dir}", file=sys.stderr)
        return 1
    main_r, other_r = rank(results)
    main_rows = [row(r, i + 1) for i, r in enumerate(main_r)]
    other_rows = [row(r, None) for r in other_r]
    md = to_markdown(main_rows, other_rows, limit)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"latency_constraint_p90_sec": limit, "main": main_rows, "other": other_rows},
                  f, ensure_ascii=False, indent=2)
    print(md)
    print(f"wrote {args.out_md}, {args.out_json}")
    if not args.no_plot:
        plot_paths(main_rows + other_rows, limit, args.out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
