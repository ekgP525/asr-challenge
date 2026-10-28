"""Build the evaluation set: clean + SNR 20/10/5/0 dB outdoor-noise versions of every source utterance.

For each utterance in data/eval/manifest_clean_*.jsonl:
  * pad 1 s (--pad) before and after the utterance; the clean condition gets digital
    silence there, noisy conditions get noise over the whole padded length, so the
    endpointer has to find the end of speech inside noise;
  * every source utterance is first scaled so its ITU-T P.56 active speech level
    equals --target-level (default -26 dBov; recordings differ by 50 dB otherwise),
    limited so the peak stays below 0.99; the applied `speech_gain` is recorded;
  * the noise level is the RMS of the noise segment actually used; noise gain is
    set so that target_level - noise_level = SNR;
  * noise category cycles over crowd/event/traffic/wind in utterance order (balanced);
    the clip and start offset are drawn from a seeded RNG per utterance
    (same noise clip/offset across the four SNRs of one utterance, so only the
    level changes). Clips shorter than needed are tiled with a 50 ms crossfade;
  * if the mixture would clip, the whole mixture is scaled down (gain recorded);
  * speech_start / speech_end are shifted by the padding.

Outputs
  data/eval/audio/{clean,snr20,snr10,snr5,snr0}/<id>.wav      16 kHz mono PCM16
  data/eval/manifest.jsonl                                     one row per file
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, SR, read_jsonl, read_wav, write_jsonl, write_wav  # noqa: E402
from p56 import active_speech_level, rms_db  # noqa: E402

SNRS = [20, 10, 5, 0]
XFADE_S = 0.05
PEAK = 0.99


def tile(noise: np.ndarray, n: int, offset: int, xfade: int) -> np.ndarray:
    """n samples of `noise` starting at `offset`; the clip is repeated with a crossfade
    at every seam (no hard cut), then the window [offset, offset+n) is taken."""
    if len(noise) <= xfade * 2:
        raise ValueError("noise clip too short")
    offset %= len(noise)
    need = n + offset
    ramp = np.linspace(0, 1, xfade, dtype=np.float32)
    out = noise
    while len(out) < need:
        seam = out[-xfade:] * (1 - ramp) + noise[:xfade] * ramp
        out = np.concatenate([out[:-xfade], seam, noise[xfade:]])
    return np.ascontiguousarray(out[offset: offset + n])


def load_noise_bank(noise_dir: str, noise_list: str):
    bank = {}
    with open(noise_list, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bank.setdefault(r["category"], []).append(r)
    for cat in bank:
        bank[cat].sort(key=lambda r: int(r["freesound_id"]))
    return bank


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--noise-dir", default=os.path.join(ROOT, "data", "noise"))
    ap.add_argument("--noise-list", default=os.path.join(ROOT, "data", "noise_list.csv"))
    ap.add_argument("--pad", type=float, default=1.0, help="seconds of noise/silence before and after")
    ap.add_argument("--target-level", type=float, default=-26.0, help="P.56 active speech level after normalization (dBov)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--snrs", default=",".join(map(str, SNRS)))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    snrs = [int(s) for s in args.snrs.split(",")]

    bank = load_noise_bank(args.noise_dir, args.noise_list)
    cats = sorted(bank)
    print("noise categories:", {c: len(bank[c]) for c in cats})
    cache = {}

    def noise_audio(rel):
        if rel not in cache:
            cache[rel] = read_wav(os.path.join(args.noise_dir, rel))
        return cache[rel]

    src_rows = []
    for p in sorted(glob.glob(os.path.join(args.eval_dir, "manifest_clean_*.jsonl"))):
        src_rows.extend(read_jsonl(p))
    if args.limit:
        src_rows = src_rows[: args.limit]
    print(f"{len(src_rows)} source utterances")

    pad = int(round(args.pad * SR))
    xfade = int(XFADE_S * SR)
    out_rows = []
    clipped = 0
    for k, u in enumerate(src_rows):
        rng = random.Random(f"{args.seed}:{u['id']}")  # per-utterance stream, order independent
        speech = read_wav(os.path.join(args.eval_dir, u["audio"]))
        raw_level, act = active_speech_level(speech, SR)
        s_gain = 10 ** ((args.target_level - raw_level) / 20)
        peak = float(np.abs(speech).max())
        if peak * s_gain > PEAK:
            s_gain = PEAK / peak
        speech = (speech * s_gain).astype(np.float32)
        s_level, _ = active_speech_level(speech, SR)
        n = len(speech) + 2 * pad
        padded = np.zeros(n, dtype=np.float32)
        padded[pad: pad + len(speech)] = speech
        base = {k2: u[k2] for k2 in ("source", "source_id", "speaker", "text")}
        base.update({"duration": round(n / SR, 3), "speech_start": round(u["speech_start"] + args.pad, 3),
                     "speech_end": round(u["speech_end"] + args.pad, 3), "seed": args.seed,
                     "speech_level_dbov": round(s_level, 2), "source_level_dbov": round(raw_level, 2),
                     "speech_gain": round(float(s_gain), 5), "activity": round(act, 3)})

        rel = f"audio/clean/{u['id']}.wav"
        write_wav(os.path.join(args.eval_dir, rel), padded)
        out_rows.append({"id": f"{u['id']}_clean", "audio": rel, "condition": "clean", "noise": None, "gain": 1.0, **base})

        cat = cats[k % len(cats)]  # balanced over categories; clip/offset still per-utterance random
        clip = rng.choice(bank[cat])
        noise = noise_audio(clip["file"])
        offset = rng.randrange(len(noise))
        seg = tile(noise, n, offset, xfade)
        n_level = rms_db(seg)
        for snr in snrs:
            g = 10 ** ((s_level - n_level - snr) / 20)
            mix = padded + g * seg
            gain = 1.0
            peak = float(np.abs(mix).max())
            if peak > PEAK:
                gain = PEAK / peak
                mix = mix * gain
                clipped += 1
            cond = f"snr{snr}"
            rel = f"audio/{cond}/{u['id']}.wav"
            write_wav(os.path.join(args.eval_dir, rel), mix)
            out_rows.append({
                "id": f"{u['id']}_{cond}", "audio": rel, "condition": cond,
                "noise": {"type": cat, "file": f"fsd50k/{clip['freesound_id']}.wav", "freesound_id": clip["freesound_id"],
                          "offset": round(offset / SR, 3), "snr_db": snr, "noise_gain": round(float(g), 5)},
                "gain": round(gain, 4), **base,
            })
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(src_rows)}", flush=True)

    out = os.path.join(args.eval_dir, "manifest.jsonl")
    m = write_jsonl(out, out_rows)
    print(f"wrote {m} rows -> {out}; {clipped} mixtures scaled to avoid clipping")


if __name__ == "__main__":
    main()
