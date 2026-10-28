"""Upload the generated evaluation audio + manifest to a Hugging Face dataset repo.

    HF_TOKEN=... python data/scripts/upload_hf.py --repo <user>/campusvoice-rt-eval

Uploads data/eval/audio/** and data/eval/manifest.jsonl, plus data/noise_list.csv
and data/LICENSES/ so the dataset card carries the attributions.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

CARD = """---
license: cc-by-4.0
language: [ko]
task_categories: [automatic-speech-recognition]
pretty_name: CampusVoice-RT evaluation set
---
# CampusVoice-RT evaluation set

Korean read speech (Zeroth-Korean test, FLEURS ko_kr test; both CC BY 4.0) mixed with
outdoor noise from FSD50K (CC0 / CC BY clips only) at SNR 20 / 10 / 5 / 0 dB, plus the
clean condition, each with 1 s of noise (or silence) before and after the utterance.
Built by https://github.com/ekgP525/asr-challenge (`data/scripts/`), seed 2026.

Files: `audio/<condition>/<id>.wav` (16 kHz mono PCM16), `manifest.jsonl`,
`noise_list.csv`, `LICENSES/`.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "eval"))
    args = ap.parse_args()
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=args.repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=os.path.join(args.eval_dir, "manifest.jsonl"), path_in_repo="manifest.jsonl",
                    repo_id=args.repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "data", "noise_list.csv"), path_in_repo="noise_list.csv",
                    repo_id=args.repo, repo_type="dataset")
    api.upload_folder(folder_path=os.path.join(ROOT, "data", "LICENSES"), path_in_repo="LICENSES",
                      repo_id=args.repo, repo_type="dataset")
    api.upload_folder(folder_path=os.path.join(args.eval_dir, "audio"), path_in_repo="audio",
                      repo_id=args.repo, repo_type="dataset", allow_patterns=["clean/*", "snr*/*"])
    print(f"uploaded to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
