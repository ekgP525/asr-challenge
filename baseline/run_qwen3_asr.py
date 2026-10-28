"""Baseline runner: Qwen3-ASR (1.7B / 0.6B), offline or streaming.

    python baseline/run_qwen3_asr.py --config baseline/configs/B1.json \
        --manifest data/eval/manifest.jsonl --audio-root data/eval --out-dir runs/B1 [--no-latency]

Writes
    <out-dir>/predictions.jsonl   {"id", "text"}                         (Track A input)
    <out-dir>/latency.jsonl       stream_sim.UttRecord per utterance      (Track B input)
    <out-dir>/environment.json    GPU / library versions

Then:
    python evaluation/evaluate.py --config baseline/configs/B1.json --manifest data/eval/manifest.jsonl \
        --predictions runs/B1/predictions.jsonl --latency runs/B1/latency.jsonl --env runs/B1/environment.json

The two adapter classes wrap the `qwen-asr` package; verify their call
signatures against https://github.com/QwenLM/Qwen3-ASR before the first run.
Pacing, VAD and timing live in evaluation/stream_sim.py and are identical for every system.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.stream_sim import (  # noqa: E402
    SileroEndpointer, load_config, measure_median, run_offline, run_streaming,
)


# --------------------------------------------------------------------------- #
# Qwen3-ASR adapters
# --------------------------------------------------------------------------- #
def _load_model(model_id: str):
    from qwen_asr import Qwen3ASRModel  # pip install -U qwen-asr[vllm]

    if hasattr(Qwen3ASRModel, "LLM"):
        return Qwen3ASRModel.LLM(model=model_id)
    return Qwen3ASRModel.from_pretrained(model_id)


def _text(out) -> str:
    if isinstance(out, list):
        out = out[0] if out else ""
    return getattr(out, "text", out if isinstance(out, str) else "")


class QwenOffline:
    def __init__(self, model_id: str):
        self.model = _load_model(model_id)

    def transcribe(self, audio, sample_rate: int) -> str:
        return _text(self.model.transcribe(audio=(audio, sample_rate), language="Korean"))


class QwenStreaming:
    def __init__(self, model_id: str):
        self.model = _load_model(model_id)
        self.state = None
        self._partial = ""

    def reset(self) -> None:
        self.state = self.model.init_streaming_state() if hasattr(self.model, "init_streaming_state") else None
        self._partial = ""

    def push(self, chunk, sample_rate: int) -> Optional[str]:
        # TODO(verify): qwen-asr streaming API name / signature.
        out = self.model.streaming_transcribe(self.state, (chunk, sample_rate), language="Korean")
        t = _text(out)
        if t:
            self._partial = t
        return self._partial or None

    def finalize(self) -> str:
        if hasattr(self.model, "finish_streaming"):
            t = _text(self.model.finish_streaming(self.state))
            if t:
                self._partial = t
        return self._partial


# --------------------------------------------------------------------------- #
def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "transformers", "vllm", "qwen_asr", "silero_vad"):
        try:
            info[name] = getattr(__import__(name), "__version__", "unknown")
        except Exception:
            info[name] = None
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_audio(path: str, target_sr: int):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(audio)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", default="data/eval")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--latency-config", default=None)
    ap.add_argument("--no-latency", action="store_true", help="Track A only: transcribe without pacing")
    ap.add_argument("--conditions", default=None, help="comma list to restrict, e.g. clean,snr0")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N utterances")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    lcfg = load_config(args.latency_config) if args.latency_config else load_config()
    sr = lcfg["sample_rate"]
    streaming = cfg["mode"] == "streaming"
    system = QwenStreaming(cfg["model"]) if streaming else QwenOffline(cfg["model"])

    utts = list(read_jsonl(args.manifest))
    if args.conditions:
        keep = set(args.conditions.split(","))
        utts = [u for u in utts if u["condition"] in keep]
    if args.limit:
        utts = utts[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(environment_info(), f, indent=2)

    pred_f = open(os.path.join(args.out_dir, "predictions.jsonl"), "w", encoding="utf-8")
    lat_f = None if args.no_latency else open(os.path.join(args.out_dir, "latency.jsonl"), "w", encoding="utf-8")
    vad = None if args.no_latency else SileroEndpointer(lcfg)

    def run_once(a):
        if streaming:
            system.reset(); system.push(a, sr); return system.finalize()
        return system.transcribe(a, sr)

    if not args.no_latency:
        for u in utts[: lcfg["warmup_utterances"]]:
            run_once(load_audio(os.path.join(args.audio_root, u["audio"]), sr))

    for i, u in enumerate(utts):
        a = load_audio(os.path.join(args.audio_root, u["audio"]), sr)
        if args.no_latency:
            text = run_once(a)
        else:
            if streaming:
                run = lambda: run_streaming(a, sr, u, system, lcfg, vad)
            else:
                run = lambda: run_offline(a, sr, u, system, lcfg, vad)
            rec = measure_median(run, lcfg["repeats"])
            text = rec.text
            lat_f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        pred_f.write(json.dumps({"id": u["id"], "text": text}, ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(utts)}", flush=True)

    pred_f.close()
    if lat_f:
        lat_f.close()
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
