"""Select clean source utterances from public Korean corpora (both CC BY 4.0).

    python data/scripts/select_subset.py --source zeroth [--n 250]
    python data/scripts/select_subset.py --source fleurs [--n 250]

zeroth  https://huggingface.co/datasets/kresnik/zeroth_korean  (parquet, 16 kHz,
        id / speaker_id / text / audio; numbers already in Hangul). Only the test
        shard (~60 MB) is downloaded.
fleurs  https://huggingface.co/datasets/google/fleurs  data/ko_kr/test.tsv + audio/test.tar.gz
        (~215 MB). Columns: sentence id, file, raw transcription, normalized
        transcription, chars, num_samples, gender. Several recordings share a
        sentence id; one recording per sentence is kept.

Selection (deterministic):
  1. test split only;
  2. keep utterances with --min-dur <= duration <= --max-dur whose reference
     contains no Arabic digits and no Latin letters (numbers would need a
     native/Sino reading decision, Latin is usually a parenthetical gloss that
     was not spoken);
  3. shuffle with --seed and pick --n utterances round-robin over speakers
     (zeroth) / one per sentence balanced over gender (fleurs);
  4. the chosen ids go to data/eval/clean_subset_ids_<source>.txt and are
     re-used verbatim on later runs (--reselect to ignore).

Outputs
  data/eval/audio/clean/<source>_XXXX.wav         16 kHz mono PCM16
  data/eval/manifest_clean_<source>.jsonl         id, audio, duration, text, source,
      source_id, speaker, condition="clean", speech_start, speech_end (Silero VAD)
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
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, SR, SpeechBounds, to_mono_float32, write_jsonl, write_wav  # noqa: E402

_BAD = re.compile(r"[0-9A-Za-z]")


# --------------------------------------------------------------------------- #
# Sources: each yields candidate dicts {id, speaker, text, duration, group} and a decode(cand) -> audio
# --------------------------------------------------------------------------- #
def load_zeroth(split: str):
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    shard = f"data/{split}-00000-of-00001.parquet"
    print(f"downloading kresnik/zeroth_korean/{shard} ...", flush=True)
    path = hf_hub_download("kresnik/zeroth_korean", shard, repo_type="dataset")
    table = pq.read_table(path, columns=["id", "speaker_id", "text", "audio"])
    ids, spk, txt = (table.column(c).to_pylist() for c in ("id", "speaker_id", "text"))
    audio_col = table.column("audio")

    def decode(c):
        a = audio_col[c["idx"]].as_py()
        audio, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        return to_mono_float32(audio, sr)

    cands = []
    for i in range(table.num_rows):
        cands.append({"idx": i, "id": ids[i], "speaker": f"zeroth_{spk[i]}", "text": txt[i].strip(),
                      "group": str(spk[i]), "duration": None})
    return cands, decode


def load_fleurs(split: str):
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    tsv = hf_hub_download("google/fleurs", f"data/ko_kr/{split}.tsv", repo_type="dataset")
    print(f"downloading google/fleurs/data/ko_kr/audio/{split}.tar.gz ...", flush=True)
    tgz = hf_hub_download("google/fleurs", f"data/ko_kr/audio/{split}.tar.gz", repo_type="dataset")
    tf = tarfile.open(tgz, "r:gz")
    members = {os.path.basename(m.name): m for m in tf.getmembers() if m.isfile()}

    def decode(c):
        f = tf.extractfile(members[c["file"]])
        audio, sr = sf.read(io.BytesIO(f.read()), dtype="float32")
        return to_mono_float32(audio, sr)

    cands = []
    with open(tsv, encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            sent_id, fname, raw, _norm, _chars, nsamp, gender = r[:7]
            if fname not in members:
                continue
            cands.append({"id": fname.replace(".wav", ""), "file": fname, "speaker": f"fleurs_{gender.lower()}",
                          "text": raw.strip().strip('"').strip(), "group": gender, "sentence_id": sent_id,
                          "duration": int(nsamp) / 16000})
    return cands, decode


SOURCES = {"zeroth": load_zeroth, "fleurs": load_fleurs}


# --------------------------------------------------------------------------- #
def select(cands, n: int, seed: int, one_per=None):
    """Seeded shuffle, optional de-duplication by `one_per` key, round-robin over `group`."""
    rng = random.Random(seed)
    rng.shuffle(cands)
    if one_per:
        seen, dedup = set(), []
        for c in cands:
            if c[one_per] not in seen:
                seen.add(c[one_per])
                dedup.append(c)
        cands = dedup
    by = defaultdict(list)
    for c in cands:
        by[c["group"]].append(c)
    groups = sorted(by)
    rng.shuffle(groups)
    out = []
    while len(out) < n and any(by[g] for g in groups):
        for g in groups:
            if by[g] and len(out) < n:
                out.append(by[g].pop())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-dur", type=float, default=1.5)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--reselect", action="store_true", help="ignore an existing ids file")
    ap.add_argument("--no-vad", action="store_true", help="skip Silero VAD (speech_start/end = 0/duration)")
    ap.add_argument("--limit", type=int, default=None, help="debug: stop after N written utterances")
    args = ap.parse_args(argv)

    src = args.source
    ids_file = os.path.join(args.out_dir, f"clean_subset_ids_{src}.txt")
    cands, decode = SOURCES[src](args.split)
    print(f"{len(cands)} utterances in {src}/{args.split}")

    fixed = None
    if not args.reselect and os.path.exists(ids_file):
        with open(ids_file, encoding="utf-8") as f:
            fixed = [l.strip() for l in f if l.strip()]
        print(f"re-using {len(fixed)} ids from {ids_file}")

    if fixed is None:
        keep = []
        for c in cands:
            if _BAD.search(c["text"]):
                continue
            if c["duration"] is None:
                c["duration"] = len(decode(c)) / SR
            if args.min_dur <= c["duration"] <= args.max_dur:
                keep.append(c)
        print(f"{len(keep)}/{len(cands)} candidates after text/duration filtering")
        chosen = select(keep, args.n, args.seed, one_per="sentence_id" if src == "fleurs" else None)
        chosen.sort(key=lambda c: c["id"])
        os.makedirs(args.out_dir, exist_ok=True)
        with open(ids_file, "w", encoding="utf-8") as f:
            f.write("\n".join(c["id"] for c in chosen) + "\n")
        print(f"selected {len(chosen)} utterances, groups {sorted({c['group'] for c in chosen})} -> {ids_file}")
    else:
        by_id = {c["id"]: c for c in cands}
        missing = [i for i in fixed if i not in by_id]
        if missing:
            sys.exit(f"{len(missing)} ids from {ids_file} not found: {missing[:5]}")
        chosen = [by_id[i] for i in fixed]

    vad = None if args.no_vad else SpeechBounds()
    rows = []
    for k, c in enumerate(chosen):
        if args.limit and k >= args.limit:
            break
        audio = decode(c)
        dur = len(audio) / SR
        uid = f"{src}_{k:04d}"
        rel = f"audio/clean/{uid}.wav"
        write_wav(os.path.join(args.out_dir, rel), audio)
        bounds = vad(audio) if vad else None
        start, end = bounds if bounds else (0.0, dur)
        rows.append({
            "id": uid, "audio": rel, "duration": round(dur, 3), "text": c["text"],
            "source": src, "source_id": c["id"], "speaker": c["speaker"],
            "condition": "clean", "speech_start": round(start, 3), "speech_end": round(end, 3),
        })
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(chosen)}", flush=True)

    out = os.path.join(args.out_dir, f"manifest_clean_{src}.jsonl")
    n = write_jsonl(out, rows)
    print(f"wrote {n} rows to {out} ({sum(r['duration'] for r in rows) / 60:.1f} min of audio)")


if __name__ == "__main__":
    main()
