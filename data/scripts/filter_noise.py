"""Pick outdoor noise clips from FSD50K (per-clip licenses) and fetch only those clips.

Source: FSD50K (Fonseca et al.), mirrored clip-by-clip on the Hugging Face Hub
as `Fhrozen/FSD50k` (clips/{dev,eval}/<freesound_id>.wav + the original
ground-truth and metadata files). Nothing else is downloaded.

Categories (AudioSet labels, clip-level ground truth; see CATEGORIES for the
exact include/exclude label sets):
  crowd    Crowd | Chatter                                   (people talking around the user)
  wind     Wind
  traffic  Traffic_noise | Car_passing_by | Bus | Motorcycle | revving
  event    Cheering | Applause                               (campus events)
Clips also labelled music, animals, water, alarms, indoor/domestic sounds etc. are
excluded; wind and traffic additionally exclude speech so the categories stay distinct.

License filter: only CC0 and CC BY (3.0) clips. CC BY-NC and Sampling+ are dropped.

Outputs
  data/noise/<category>/<freesound_id>.wav     16 kHz mono PCM16 (git-ignored)
  data/noise_list.csv                          file, category, split, freesound_id, duration,
                                               title, uploader, license, source_url
  data/LICENSES/FSD50K.md                      attribution text (generated)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, SR, to_mono_float32, write_wav  # noqa: E402

REPO = "Fhrozen/FSD50k"
OK_LICENSES = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0 1.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC BY 3.0",
    "http://creativecommons.org/licenses/by/4.0/": "CC BY 4.0",
}
_NOT_HUMAN_OUTDOOR = {"Music", "Musical_instrument", "Singing", "Animal", "Bird", "Wild_animals",
                      "Domestic_animals_and_pets", "Domestic_sounds_and_home_sounds", "Alarm", "Siren",
                      "Bell", "Race_car_and_auto_racing", "Water", "Ocean", "Rain", "Thunder", "Thunderstorm", "Liquid",
                      "Drum", "Drum_kit", "Percussion", "Respiratory_sounds"}
CATEGORIES = {
    "crowd": {"any": {"Crowd", "Chatter"},
              "exclude": _NOT_HUMAN_OUTDOOR | {"Vehicle", "Engine", "Motor_vehicle_(road)", "Applause", "Cheering"}},
    "wind": {"any": {"Wind"},
             "exclude": _NOT_HUMAN_OUTDOOR | {"Speech", "Human_voice", "Vehicle", "Engine", "Mechanisms",
                                              "Wind_chime", "Wind_instrument_and_woodwind_instrument", "Respiratory_sounds"}},
    "traffic": {"any": {"Traffic_noise_and_roadway_noise", "Car_passing_by", "Bus", "Motorcycle",
                        "Accelerating_and_revving_and_vroom"},
                "exclude": _NOT_HUMAN_OUTDOOR | {"Speech", "Human_voice", "Engine_starting"}},
    "event": {"any": {"Cheering", "Applause"}, "exclude": _NOT_HUMAN_OUTDOOR},
}


def load_metadata():
    from huggingface_hub import hf_hub_download

    gt, info = [], {}
    for split in ("dev", "eval"):
        p = hf_hub_download(REPO, f"labels/{split}.csv", repo_type="dataset")
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                gt.append({"fname": r["fname"], "labels": set(r["labels"].split(",")), "split": split})
        p = hf_hub_download(REPO, f"metadata/{split}_clips_info_FSD50K.json", repo_type="dataset")
        with open(p, encoding="utf-8") as f:
            info.update(json.load(f))
    return gt, info


def candidates(gt, info):
    out = {c: [] for c in CATEGORIES}
    for r in gt:
        lic = info[r["fname"]]["license"]
        if lic not in OK_LICENSES:
            continue
        for cat, spec in CATEGORIES.items():
            if r["labels"] & spec["any"] and not (r["labels"] & spec["exclude"]):
                out[cat].append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-category", type=int, default=40)
    ap.add_argument("--min-dur", type=float, default=4.0, help="drop clips shorter than this (s)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "noise"))
    ap.add_argument("--list", default=os.path.join(ROOT, "data", "noise_list.csv"))
    ap.add_argument("--license-md", default=os.path.join(ROOT, "data", "LICENSES", "FSD50K.md"))
    args = ap.parse_args(argv)

    import soundfile as sf
    from huggingface_hub import hf_hub_download

    gt, info = load_metadata()
    cands = candidates(gt, info)
    rng = random.Random(args.seed)
    rows = []
    for cat, lst in cands.items():
        lst = sorted(lst, key=lambda r: int(r["fname"]))
        rng.shuffle(lst)
        kept = 0
        print(f"[{cat}] {len(lst)} licensed candidates", flush=True)
        for r in lst:
            if kept >= args.per_category:
                break
            fid = r["fname"]
            try:
                p = hf_hub_download(REPO, f"clips/{r['split']}/{fid}.wav", repo_type="dataset")
            except Exception as e:  # missing clip in the mirror
                print(f"  skip {fid}: {e.__class__.__name__}", flush=True)
                continue
            audio, sr = sf.read(p, dtype="float32")
            audio = to_mono_float32(audio, sr)
            dur = len(audio) / SR
            if dur < args.min_dur or float(abs(audio).max()) < 1e-4:
                continue
            rel = f"{cat}/{fid}.wav"
            write_wav(os.path.join(args.out_dir, rel), audio)
            m = info[fid]
            rows.append({
                "file": rel, "category": cat, "split": r["split"], "freesound_id": fid,
                "duration": round(dur, 3), "labels": "|".join(sorted(r["labels"])),
                "title": m["title"], "uploader": m["uploader"], "license": OK_LICENSES[m["license"]],
                "license_url": m["license"], "source_url": f"https://freesound.org/s/{fid}/",
            })
            kept += 1
        print(f"[{cat}] kept {kept}", flush=True)

    os.makedirs(os.path.dirname(args.list), exist_ok=True)
    with open(args.list, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    os.makedirs(os.path.dirname(args.license_md), exist_ok=True)
    with open(args.license_md, "w", encoding="utf-8") as f:
        f.write("# FSD50K noise clips used in CampusVoice-RT\n\n")
        f.write("FSD50K: Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra. "
                "\"FSD50K: an Open Dataset of Human-Labeled Sound Events\", IEEE/ACM TASLP 2022. "
                "https://zenodo.org/record/4060432 (dataset CC BY 4.0; clips carry their own license).\n\n")
        f.write("Only CC0 and CC BY clips are used. Each clip is attributed below "
                "(Freesound id, title, uploader, license). Clips were converted to 16 kHz mono "
                "and mixed into speech at fixed SNRs; see data/scripts/mix_noise.py.\n\n")
        f.write("| category | Freesound id | title | uploader | license |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['category']} | [{r['freesound_id']}]({r['source_url']}) | {r['title']} | "
                    f"{r['uploader']} | [{r['license']}]({r['license_url']}) |\n")
    total = sum(r["duration"] for r in rows)
    print(f"wrote {len(rows)} clips ({total / 60:.1f} min) -> {args.list}, {args.license_md}")


if __name__ == "__main__":
    main()
