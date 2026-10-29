#!/bin/bash
# Run the baselines end to end on Colab and keep progress on Drive.
#   SYSTEMS="B2 B3 B4" nohup bash scripts/colab_run_all.sh > overnight.log 2>&1 &
# Re-running after a crash continues from Drive: finished systems are skipped,
# an interrupted system resumes from its partial predictions/latency files.
set -u
cd "$(dirname "$0")/.."
SYSTEMS="${SYSTEMS:-B1 B2 B3 B4}"
SAVE="${SAVE:-/content/drive/MyDrive/campusvoice-rt}"
HW="${HW:-NVIDIA L4}"
mkdir -p "$SAVE/runs" "$SAVE/results" runs results

sync_to_drive() { cp -r runs/. "$SAVE/runs/" 2>/dev/null; cp results/*.json "$SAVE/results/" 2>/dev/null; }
# background: copy partial progress every 10 minutes
( while true; do sleep 600; sync_to_drive; echo "[sync $(date +%H:%M)]"; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null' EXIT

for sid in $SYSTEMS; do
  if [ -f "$SAVE/results/$sid.json" ]; then echo "== $sid already done (Drive)"; cp "$SAVE/results/$sid.json" results/; cp -r "$SAVE/runs/$sid" runs/ 2>/dev/null; continue; fi
  if [ -d "$SAVE/runs/$sid" ] && [ ! -f "runs/$sid/predictions.jsonl" ]; then cp -r "$SAVE/runs/$sid" runs/; fi
  echo "== $sid start $(date)"
  python baseline/run_qwen3_asr.py --config baseline/configs/$sid.json --manifest data/eval/manifest.jsonl \
      --audio-root data/eval --out-dir runs/$sid --resume || { echo "== $sid FAILED"; sync_to_drive; exit 1; }
  python evaluation/evaluate.py --config baseline/configs/$sid.json --manifest data/eval/manifest.jsonl \
      --predictions runs/$sid/predictions.jsonl --latency runs/$sid/latency.jsonl \
      --env runs/$sid/environment.json --hardware "$HW" || exit 1
  sync_to_drive
  echo "== $sid done $(date)"
done

if [ ! -f "$SAVE/results/jitter_B2.json" ] && [ -f runs/B2/latency.jsonl ]; then
  echo "== jitter check start $(date)"
  python baseline/run_qwen3_asr.py --config baseline/configs/B2.json --manifest data/eval/manifest.jsonl \
      --audio-root data/eval --out-dir runs/jitter_B2 --limit 200 --repeats 3 --resume
  python evaluation/jitter_check.py --single runs/B2/latency.jsonl --repeated runs/jitter_B2/latency.jsonl --out results/jitter_B2.json
  sync_to_drive
fi
echo ALL_DONE
