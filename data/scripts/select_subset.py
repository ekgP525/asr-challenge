"""Select clean source utterances from zeroth-korean (CC BY 4.0).

Source: https://huggingface.co/datasets/kresnik/zeroth_korean  (parquet, 16 kHz,
fields id / speaker_id / chapter_id / audio / text; numbers already in Hangul).
Only the test shard (~60 MB) is downloaded via huggingface_hub; `datasets` is not needed.

Selection (deterministic):
  1. take the `test` split (457 utterances, 10 speakers);
  2. keep utterances with --min-dur <= duration <= --max-dur, whose text is
     stable under evaluation.normalizer (no digits / symbols to resolve);
  3. shuffle with --seed and pick --n utterances round-robin over speakers so
     no single speaker dominates;
  4. the chosen ids are written to data/eval/clean_subset_ids.txt and re-used
     verbatim on later runs (pass --reselect to ignore that file).

Outputs
  data/eval/audio/clean/zeroth_XXXX.wav           16 kHz mono PCM16
  data/eval/manifest_clean_zeroth.jsonl           one row per utterance:
      id, audio, duration, text, source="zeroth", source_id, speaker,
      condition="clean", speech_start, speech_end
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, SR, SpeechBounds, to_mono_float32, write_jsonl, write_wav  # noqa: E402
from evaluation.normalizer import normalize  # noqa: E402

HF_DATASET = "kresnik/zeroth_korean"


def select(rows, n: int, seed: int):
    """Round-robin over speakers after a seeded shuffle -> balanced, deterministic."""
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["speaker_id"]].append(r)
    for lst in by_spk.values():
        rng.shuffle(lst)
    speakers = sorted(by_spk)
    rng.shuffle(speakers)
    out = []
    while len(out) < n and any(by_spk[s] for s in speakers):
        for s in speakers:
            if by_spk[s] and len(out) < n:
                out.append(by_spk[s].pop())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-dur", type=float, default=1.5)
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--ids-file", default=None, help="default <out-dir>/clean_subset_ids.txt")
    ap.add_argument("--reselect", action="store_true", help="ignore an existing ids file")
    ap.add_argument("--no-vad", action="store_true", help="skip Silero VAD (speech_start/end = 0/duration)")
    ap.add_argument("--limit", type=int, default=None, help="debug: stop after N written utterances")
    args = ap.parse_args(argv)

    import io
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    ids_file = args.ids_file or os.path.join(args.out_dir, "clean_subset_ids.txt")

    # Download only the requested split's parquet shard (the full dataset is ~3 GB).
    shard = f"data/{args.split}-00000-of-00001.parquet"
    print(f"downloading {HF_DATASET}/{shard} ...", flush=True)
    path = hf_hub_download(HF_DATASET, shard, repo_type="dataset")
    table = pq.read_table(path, columns=["id", "speaker_id", "text", "audio"])
    cols = {c: table.column(c).to_pylist() for c in ("id", "speaker_id", "text")}
    audio_col = table.column("audio")
    meta = [{"idx": i, "id": cols["id"][i], "speaker_id": cols["speaker_id"][i], "text": cols["text"][i]}
            for i in range(table.num_rows)]
    print(f"{len(meta)} utterances in split '{args.split}'")

    def decode(i):
        a = audio_col[i].as_py()
        audio, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        return to_mono_float32(audio, sr)

    fixed_ids = None
    if not args.reselect and os.path.exists(ids_file):
        with open(ids_file, encoding="utf-8") as f:
            fixed_ids = [l.strip() for l in f if l.strip()]
        print(f"re-using {len(fixed_ids)} ids from {ids_file}")

    if fixed_ids is None:
        # filter by duration + normalization stability, then select
        cands = []
        for m in meta:
            txt = m["text"].strip()
            if normalize(txt) != txt:
                continue
            audio = decode(m["idx"])
            dur = len(audio) / SR
            if args.min_dur <= dur <= args.max_dur:
                cands.append({**m, "duration": dur})
        print(f"{len(cands)}/{len(meta)} candidates after filtering")
        chosen = select(cands, args.n, args.seed)
        chosen.sort(key=lambda r: r["id"])
        os.makedirs(args.out_dir, exist_ok=True)
        with open(ids_file, "w", encoding="utf-8") as f:
            f.write("\n".join(r["id"] for r in chosen) + "\n")
        print(f"selected {len(chosen)} utterances from {len({r['speaker_id'] for r in chosen})} speakers -> {ids_file}")
    else:
        by_id = {m["id"]: m for m in meta}
        missing = [i for i in fixed_ids if i not in by_id]
        if missing:
            sys.exit(f"{len(missing)} ids from {ids_file} not found in {HF_DATASET}: {missing[:5]}")
        chosen = [by_id[i] for i in fixed_ids]

    vad = None if args.no_vad else SpeechBounds()
    rows = []
    for k, m in enumerate(chosen):
        if args.limit and k >= args.limit:
            break
        audio = decode(m["idx"])
        dur = len(audio) / SR
        uid = f"zeroth_{k:04d}"
        rel = f"audio/clean/{uid}.wav"
        write_wav(os.path.join(args.out_dir, rel), audio)
        bounds = vad(audio) if vad else None
        start, end = bounds if bounds else (0.0, dur)
        text = m["text"].strip()
        rows.append({
            "id": uid, "audio": rel, "duration": round(dur, 3), "text": text,
            "source": "zeroth",
            "source_id": m["id"], "speaker": f"zeroth_{m['speaker_id']}",
            "condition": "clean", "speech_start": round(start, 3), "speech_end": round(end, 3),
        })
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(chosen)}", flush=True)

    out = os.path.join(args.out_dir, "manifest_clean_zeroth.jsonl")
    n = write_jsonl(out, rows)
    total = sum(r["duration"] for r in rows)
    print(f"wrote {n} rows to {out} ({total / 60:.1f} min of audio)")


if __name__ == "__main__":
    main()
