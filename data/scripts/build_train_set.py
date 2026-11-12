"""Build the S3 training set: public Korean read speech + outdoor noise, disjoint from the evaluation set.

    python data/scripts/build_train_set.py noise  [--per-category 150]           # training noise bank
    python data/scripts/build_train_set.py speech [--workers 8] [--limit N]       # mixtures + jsonl

Speech: kresnik/zeroth_korean *train* (6 parquet shards, ~22k utterances, 51 h, CC BY 4.0) and
google/fleurs ko_kr *train* (2,307 recordings, CC BY 4.0). The evaluation set uses only the *test*
splits of both, so speakers/sentences do not overlap. Same text filter as the evaluation set (no
Arabic digits, no Latin letters) and 1.5-20 s duration.
Noise: FSD50K clips of the same four categories (crowd / wind / traffic / event) with the same license
filter, *excluding* the 160 clips used in the evaluation set (data/noise_list.csv).
Dev: google/fleurs ko_kr *dev* (226 recordings) x {clean, 20, 10, 5, 0 dB} with the evaluation
set's 1 s padding - for CER during/after training; never the evaluation set itself.

Mixing (per utterance, seeded by id): P.56 active speech level normalized to -26 dBov like the eval
set; leading pad U(0.1, 0.6) s and trailing pad U(0.3, 1.0) s (what the ASR receives in the
simulator: 200 ms pre-roll to a few hundred ms after speech); with probability --clean-frac the
file is clean with the -60 dBov microphone floor, otherwise a random category/clip/offset at
SNR ~ U(--snr-min, --snr-max) dB. Clipping is avoided by scaling the mixture.

Outputs (git-ignored audio)
  data/noise_train/<cat>/<id>.wav, data/noise_train_list.csv, data/LICENSES/FSD50K_train.md
  data/train/audio/<id>.wav, data/train/train.jsonl  ({"audio", "text": "language Korean<asr_text>..."})
  data/train/train_manifest.jsonl (id, audio, text, source, speaker, condition, snr_db, noise, ...)
  data/train/dev/audio/<id>.wav, data/train/dev.jsonl, data/train/dev_manifest.jsonl
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import re
import sys
import tarfile
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, SR, read_jsonl, read_wav, to_mono_float32, write_jsonl, write_wav  # noqa: E402
from mix_noise import PEAK, XFADE_S, tile  # noqa: E402
from p56 import active_speech_level, rms_db  # noqa: E402

_BAD = re.compile(r"[0-9A-Za-z]")
TARGET_LEVEL = -26.0
CLEAN_FLOOR = -60.0
SFT_PREFIX = "language Korean<asr_text>"
DEV_SNRS = [None, 20, 10, 5, 0]           # None = clean


# --------------------------------------------------------------------------- #
# noise bank for training (disjoint from the evaluation clips)
# --------------------------------------------------------------------------- #
def build_noise_bank(per_category: int, min_dur: float, seed: int, out_dir: str, list_path: str,
                     license_md: str, exclude_list: str):
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from filter_noise import OK_LICENSES, REPO, candidates, load_metadata

    used = set()
    if os.path.exists(exclude_list):
        with open(exclude_list, encoding="utf-8") as f:
            used = {r["freesound_id"] for r in csv.DictReader(f)}
    gt, info = load_metadata()
    cands = candidates(gt, info)
    rng = random.Random(seed)
    rows = []
    for cat, lst in cands.items():
        lst = sorted([r for r in lst if r["fname"] not in used], key=lambda r: int(r["fname"]))
        rng.shuffle(lst)
        kept = 0
        print(f"[{cat}] {len(lst)} licensed candidates not in the evaluation set", flush=True)
        for r in lst:
            if kept >= per_category:
                break
            fid = r["fname"]
            try:
                p = hf_hub_download(REPO, f"clips/{r['split']}/{fid}.wav", repo_type="dataset")
            except Exception as e:
                print(f"  skip {fid}: {e.__class__.__name__}", flush=True)
                continue
            audio, sr = sf.read(p, dtype="float32")
            audio = to_mono_float32(audio, sr)
            dur = len(audio) / SR
            if dur < min_dur or float(abs(audio).max()) < 1e-4:
                continue
            rel = f"{cat}/{fid}.wav"
            write_wav(os.path.join(out_dir, rel), audio)
            m = info[fid]
            rows.append({"file": rel, "category": cat, "split": r["split"], "freesound_id": fid,
                         "duration": round(dur, 3), "labels": "|".join(sorted(r["labels"])),
                         "title": m["title"], "uploader": m["uploader"], "license": OK_LICENSES[m["license"]],
                         "license_url": m["license"], "source_url": f"https://freesound.org/s/{fid}/"})
            kept += 1
        print(f"[{cat}] kept {kept}", flush=True)
    os.makedirs(os.path.dirname(list_path), exist_ok=True)
    with open(list_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    os.makedirs(os.path.dirname(license_md), exist_ok=True)
    with open(license_md, "w", encoding="utf-8") as f:
        f.write("# FSD50K noise clips used for *training* CampusVoice-RT participant systems\n\n")
        f.write("Disjoint from the evaluation clips (data/LICENSES/FSD50K.md). Only CC0 and CC BY clips.\n\n")
        f.write("| category | Freesound id | title | uploader | license |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['category']} | [{r['freesound_id']}]({r['source_url']}) | {r['title']} | "
                    f"{r['uploader']} | [{r['license']}]({r['license_url']}) |\n")
    print(f"wrote {len(rows)} clips ({sum(r['duration'] for r in rows) / 60:.1f} min) -> {list_path}")


# --------------------------------------------------------------------------- #
# mixing
# --------------------------------------------------------------------------- #
def normalize_speech(speech: np.ndarray):
    raw_level, act = active_speech_level(speech, SR)
    if not np.isfinite(raw_level):
        return None
    g = 10 ** ((TARGET_LEVEL - raw_level) / 20)
    peak = float(np.abs(speech).max())
    if peak * g > PEAK:
        g = PEAK / peak
    out = (speech * g).astype(np.float32)
    level, _ = active_speech_level(out, SR)
    return out, level, raw_level, float(g), act


def mix_one(speech: np.ndarray, noise: np.ndarray | None, snr_db: float | None, lead_s: float, trail_s: float,
            rng: random.Random, floor_seed: int):
    """-> (audio, info). `noise` None or `snr_db` None = clean with the microphone floor."""
    norm = normalize_speech(speech)
    if norm is None:
        return None, None
    speech, s_level, raw_level, s_gain, act = norm
    lead, trail = int(round(lead_s * SR)), int(round(trail_s * SR))
    n = lead + len(speech) + trail
    padded = np.zeros(n, dtype=np.float32)
    padded[lead: lead + len(speech)] = speech
    info = {"duration": round(n / SR, 3), "speech_start": round(lead / SR, 3), "speech_end": round((lead + len(speech)) / SR, 3),
            "speech_level_dbov": round(s_level, 2), "source_level_dbov": round(raw_level, 2), "speech_gain": round(s_gain, 5),
            "activity": round(act, 3), "gain": 1.0}
    if noise is None or snr_db is None:
        floor = np.random.default_rng(floor_seed).standard_normal(n).astype(np.float32) * 10 ** (CLEAN_FLOOR / 20)
        info.update({"condition": "clean", "snr_db": round(s_level - CLEAN_FLOOR, 1)})
        return padded + floor, info
    offset = rng.randrange(len(noise))
    seg = tile(noise, n, offset, int(XFADE_S * SR))
    g = 10 ** ((s_level - rms_db(seg) - snr_db) / 20)
    mix = padded + g * seg
    peak = float(np.abs(mix).max())
    if peak > PEAK:
        info["gain"] = round(PEAK / peak, 4)
        mix = mix * (PEAK / peak)
    info.update({"condition": "noisy", "snr_db": round(snr_db, 2), "noise_offset": round(offset / SR, 3), "noise_gain": round(float(g), 5)})
    return mix.astype(np.float32), info


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #
_W = {}


def _init(noise_dir, noise_list, args_d):
    bank = {}
    with open(noise_list, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bank.setdefault(r["category"], []).append(r)
    for c in bank:
        bank[c].sort(key=lambda r: int(r["freesound_id"]))
    _W.update({"bank": bank, "cats": sorted(bank), "noise_dir": noise_dir, "cache": {}, "args": args_d})


def _noise(rel):
    c = _W["cache"]
    if rel not in c:
        c[rel] = read_wav(os.path.join(_W["noise_dir"], rel))
    return c[rel]


def _job(task):
    """task: dict(uid, wav_bytes, text, source, speaker, out_dir, mode, snr(None|float), pads(None|tuple))"""
    import soundfile as sf
    a = _W["args"]
    try:
        audio, sr = sf.read(io.BytesIO(task["wav_bytes"]), dtype="float32")
    except Exception:
        return None
    audio = to_mono_float32(audio, sr)
    dur = len(audio) / SR
    if not (a["min_dur"] <= dur <= a["max_dur"]):
        return None
    rng = random.Random(f"{a['seed']}:{task['uid']}")
    floor_seed = random.Random(f"{a['seed']}:{task['uid']}:floor").getrandbits(32)
    if task["mode"] == "dev":
        lead_s, trail_s = 1.0, 1.0
        snr = task["snr"]
    else:
        lead_s, trail_s = rng.uniform(0.1, 0.6), rng.uniform(0.3, 1.0)
        snr = None if rng.random() < a["clean_frac"] else rng.uniform(a["snr_min"], a["snr_max"])
    noise, clip = None, None
    if snr is not None:
        cat = rng.choice(_W["cats"])
        clip = rng.choice(_W["bank"][cat])
        noise = _noise(clip["file"])
    out, info = mix_one(audio, noise, snr, lead_s, trail_s, rng, floor_seed)
    if out is None:
        return None
    rel = f"audio/{task['uid']}.wav"
    write_wav(os.path.join(task["out_dir"], rel), out)
    row = {"id": task["uid"], "audio": rel, "text": task["text"], "source": task["source"], "speaker": task["speaker"],
           "source_duration": round(dur, 3), **info}
    if clip is not None:
        row["noise"] = {"type": clip["category"], "file": clip["file"], "freesound_id": clip["freesound_id"]}
    return row


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def iter_zeroth_train(shards, limit=None):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    n = 0
    for s in shards:
        name = f"data/train-{s:05d}-of-00006.parquet"
        print(f"downloading kresnik/zeroth_korean/{name} ...", flush=True)
        path = hf_hub_download("kresnik/zeroth_korean", name, repo_type="dataset")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["id", "speaker_id", "text", "audio"]):
            d = batch.to_pydict()
            for i in range(len(d["id"])):
                text = d["text"][i].strip()
                if _BAD.search(text):
                    continue
                yield {"uid": f"zeroth_train_{d['id'][i]}", "wav_bytes": d["audio"][i]["bytes"], "text": text,
                       "source": "zeroth", "speaker": f"zeroth_{d['speaker_id'][i]}"}
                n += 1
                if limit and n >= limit:
                    return


def iter_fleurs(split, limit=None):
    from huggingface_hub import hf_hub_download
    tsv = hf_hub_download("google/fleurs", f"data/ko_kr/{split}.tsv", repo_type="dataset")
    print(f"downloading google/fleurs/data/ko_kr/audio/{split}.tar.gz ...", flush=True)
    tgz = hf_hub_download("google/fleurs", f"data/ko_kr/audio/{split}.tar.gz", repo_type="dataset")
    tf = tarfile.open(tgz, "r:gz")
    members = {os.path.basename(m.name): m for m in tf.getmembers() if m.isfile()}
    n = 0
    with open(tsv, encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            sent_id, fname, raw, _norm, _chars, nsamp, gender = r[:7]
            text = raw.strip().strip('"').strip()
            if fname not in members or _BAD.search(text):
                continue
            yield {"uid": f"fleurs_{split}_{fname.replace('.wav', '')}", "wav_bytes": tf.extractfile(members[fname]).read(),
                   "text": text, "source": "fleurs", "speaker": f"fleurs_{gender.lower()}"}
            n += 1
            if limit and n >= limit:
                return


# --------------------------------------------------------------------------- #
def run_pool(tasks, out_dir, workers, noise_dir, noise_list, args_d, log_every=500):
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    rows = []
    with Pool(workers, initializer=_init, initargs=(noise_dir, noise_list, args_d)) as pool:
        for k, row in enumerate(pool.imap_unordered(_job, tasks, chunksize=8)):
            if row:
                rows.append(row)
            if (k + 1) % log_every == 0:
                print(f"  {k + 1} processed, {len(rows)} kept", flush=True)
    rows.sort(key=lambda r: r["id"])
    return rows


def write_sft(rows, out_dir, name):
    sft = [{"audio": os.path.abspath(os.path.join(out_dir, r["audio"])), "text": SFT_PREFIX + r["text"]} for r in rows]
    write_jsonl(os.path.join(out_dir, f"{name}.jsonl"), sft)
    write_jsonl(os.path.join(out_dir, f"{name}_manifest.jsonl"), rows)
    hours = sum(r["duration"] for r in rows) / 3600
    cond = {}
    for r in rows:
        cond[r["condition"]] = cond.get(r["condition"], 0) + 1
    print(f"{name}: {len(rows)} files, {hours:.1f} h, {cond} -> {out_dir}/{name}.jsonl")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["noise", "speech"])
    ap.add_argument("--per-category", type=int, default=150)
    ap.add_argument("--min-noise-dur", type=float, default=4.0)
    ap.add_argument("--noise-dir", default=os.path.join(ROOT, "data", "noise_train"))
    ap.add_argument("--noise-list", default=os.path.join(ROOT, "data", "noise_train_list.csv"))
    ap.add_argument("--exclude-list", default=os.path.join(ROOT, "data", "noise_list.csv"))
    ap.add_argument("--license-md", default=os.path.join(ROOT, "data", "LICENSES", "FSD50K_train.md"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "train"))
    ap.add_argument("--zeroth-shards", default="0-5", help="e.g. 0-5, 0,1 or none")
    ap.add_argument("--no-fleurs", action="store_true")
    ap.add_argument("--no-dev", action="store_true")
    ap.add_argument("--clean-frac", type=float, default=0.15)
    ap.add_argument("--snr-min", type=float, default=-3.0)
    ap.add_argument("--snr-max", type=float, default=20.0)
    ap.add_argument("--min-dur", type=float, default=1.5)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None, help="debug: utterances per source")
    args = ap.parse_args(argv)

    if args.stage == "noise":
        build_noise_bank(args.per_category, args.min_noise_dur, args.seed, args.noise_dir, args.noise_list,
                         args.license_md, args.exclude_list)
        return

    if not os.path.exists(args.noise_list):
        sys.exit(f"{args.noise_list} missing: run the 'noise' stage first")
    args_d = {k: getattr(args, k) for k in ("seed", "clean_frac", "snr_min", "snr_max", "min_dur", "max_dur")}
    shards = []
    if args.zeroth_shards != "none":
        if "-" in args.zeroth_shards:
            a, b = args.zeroth_shards.split("-"); shards = list(range(int(a), int(b) + 1))
        else:
            shards = [int(x) for x in args.zeroth_shards.split(",")]

    def train_tasks():
        for t in iter_zeroth_train(shards, args.limit):
            yield {**t, "out_dir": args.out_dir, "mode": "train", "snr": None}
        if not args.no_fleurs:
            for t in iter_fleurs("train", args.limit):
                yield {**t, "out_dir": args.out_dir, "mode": "train", "snr": None}

    rows = run_pool(train_tasks(), args.out_dir, args.workers, args.noise_dir, args.noise_list, args_d)
    write_sft(rows, args.out_dir, "train")

    if not args.no_dev:
        dev_dir = os.path.join(args.out_dir, "dev")

        def dev_tasks():
            for t in iter_fleurs("dev", args.limit):
                for snr in DEV_SNRS:
                    cond = "clean" if snr is None else f"snr{snr}"
                    yield {**t, "uid": f"{t['uid']}_{cond}", "out_dir": dev_dir, "mode": "dev", "snr": snr}

        drows = run_pool(dev_tasks(), dev_dir, args.workers, args.noise_dir, args.noise_list, args_d, log_every=200)
        for r in drows:
            r["condition"] = r["id"].rsplit("_", 1)[1]        # clean / snr20 / ... (eval-style, for evaluate.py)
        write_sft(drows, dev_dir, "dev")


if __name__ == "__main__":
    main()
