#!/bin/bash
# Fine-tune Qwen3-ASR on the CampusVoice-RT training set (Colab A100) and publish the checkpoint.
#   MODEL=Qwen/Qwen3-ASR-0.6B HF_REPO=<user>/campusvoice-rt-s3-0.6b nohup bash scripts/colab_finetune.sh > ft.log 2>&1 &
# Prerequisites: data/train/train.jsonl + dev.jsonl (data/scripts/build_train_set.py), `huggingface-cli login`
# (write token) so the checkpoint can be uploaded, qwen-asr installed.
# Re-running resumes from the last checkpoint in $OUT (--resume 1).
set -eu
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen3-ASR-0.6B}"
OUT="${OUT:-/content/ft/$(basename "$MODEL")-s3}"
HF_REPO="${HF_REPO:-}"                 # e.g. ekgP115/campusvoice-rt-s3-0.6b (private); empty = no upload
TRAIN="${TRAIN:-data/train/train.jsonl}"
DEV="${DEV:-data/train/dev/dev.jsonl}"
BS="${BS:-16}"; GA="${GA:-4}"; LR="${LR:-1e-5}"; EPOCHS="${EPOCHS:-2}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SFT_DIR="${SFT_DIR:-/content/Qwen3-ASR}"

if [ ! -f "$SFT_DIR/finetuning/qwen3_asr_sft.py" ]; then
  git clone --depth 1 https://github.com/QwenLM/Qwen3-ASR.git "$SFT_DIR"
fi
python -c "import flash_attn" 2>/dev/null || echo "note: flash-attn not installed; training uses the default attention (slower, more memory)"
mkdir -p "$OUT"
echo "== train $(date) model=$MODEL out=$OUT bs=$BS ga=$GA lr=$LR epochs=$EPOCHS"
python "$SFT_DIR/finetuning/qwen3_asr_sft.py" \
  --model_path "$MODEL" --train_file "$TRAIN" --eval_file "$DEV" --output_dir "$OUT" \
  --batch_size "$BS" --grad_acc "$GA" --lr "$LR" --epochs "$EPOCHS" \
  --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit 2 --log_steps 10 \
  --num_workers 4 --resume 1
LAST=$(ls -d "$OUT"/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "== last checkpoint: $LAST"
# quick sanity: offline transcription of one dev file with the fine-tuned weights
python - "$LAST" "$DEV" <<'PY'
import json, sys, torch
from qwen_asr import Qwen3ASRModel
ckpt, dev = sys.argv[1], sys.argv[2]
row = json.loads(open(dev, encoding="utf-8").readline())
m = Qwen3ASRModel.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda:0", max_new_tokens=256)
print("ref:", row["text"].split("<asr_text>")[-1]); print("hyp:", m.transcribe(audio=row["audio"], language="Korean")[0].text)
PY
if [ -n "$HF_REPO" ]; then
  echo "== upload $LAST -> $HF_REPO (private)"
  python - "$LAST" "$HF_REPO" <<'PY'
import sys
from huggingface_hub import HfApi
ckpt, repo = sys.argv[1], sys.argv[2]
api = HfApi(); api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path=ckpt, repo_id=repo, ignore_patterns=["optimizer.pt", "scheduler.pt", "rng_state*.pth", "trainer_state.json", "training_args.bin"])
print("uploaded", repo)
PY
fi
echo FT_DONE
