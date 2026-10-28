"""Upload the generated evaluation audio + manifest to a Hugging Face dataset repo.

    HF_TOKEN=... python data/scripts/upload_hf.py --repo ekgP115/campusvoice-rt-eval

Uploads data/eval/manifest.jsonl, data/noise_list.csv, data/LICENSES/ and the audio
both as individual files (audio/<cond>/<id>.wav, browsable on the Hub) and as a
single `audio.tar.gz`. Downloading 2,125 small files trips the Hub's API rate limit
(1,000 requests / 5 min); the tarball is one request, so notebooks fetch that.
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

Files: `audio.tar.gz` (all audio in one archive; download this), `audio/<condition>/<id>.wav`
(same files, browsable), `manifest.jsonl`, `noise_list.csv`, `LICENSES/`.

```bash
huggingface-cli download ekgP115/campusvoice-rt-eval --repo-type dataset \
    --include manifest.jsonl audio.tar.gz --local-dir data/eval && tar -xzf data/eval/audio.tar.gz -C data/eval
```
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--tar-only", action="store_true", help="only (re)upload audio.tar.gz + metadata")
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
    if not args.tar_only:
        api.upload_folder(folder_path=os.path.join(args.eval_dir, "audio"), path_in_repo="audio",
                          repo_id=args.repo, repo_type="dataset", allow_patterns=["clean/*", "snr*/*"])
    import tarfile
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "audio.tar.gz")
        with tarfile.open(tar_path, "w:gz", compresslevel=1) as tf:
            for cond in ("clean", "snr20", "snr10", "snr5", "snr0"):
                tf.add(os.path.join(args.eval_dir, "audio", cond), arcname=f"audio/{cond}")
        print(f"audio.tar.gz: {os.path.getsize(tar_path) / 1e6:.0f} MB")
        api.upload_file(path_or_fileobj=tar_path, path_in_repo="audio.tar.gz", repo_id=args.repo, repo_type="dataset")
    print(f"uploaded to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
