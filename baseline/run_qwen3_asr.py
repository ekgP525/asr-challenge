"""Baseline runner: Qwen3-ASR (1.7B / 0.6B), offline or streaming, through the shared simulator.

    python baseline/run_qwen3_asr.py --config baseline/configs/B1.json \
        --manifest data/eval/manifest.jsonl --audio-root data/eval --out-dir runs/B1

Writes
    <out-dir>/predictions.jsonl   {"id", "text"}                       (Track A input)
    <out-dir>/latency.jsonl       stream_sim.UttRecord per utterance    (Track B input)
    <out-dir>/environment.json    GPU / library versions

Config keys (baseline/configs/*.json)
    model                    HF id, e.g. Qwen/Qwen3-ASR-1.7B
    mode                     "offline" | "streaming"
    backend                  "vllm" (default; required for streaming) | "transformers"
    gpu_memory_utilization   vLLM only
    max_new_tokens
    streaming.chunk_sec      model-side decode granularity (the simulator always feeds 0.5 s)
    streaming.unfixed_chunk_num / unfixed_token_num   qwen-asr prefix-rollback parameters

qwen-asr 0.0.6 API used here:
    Qwen3ASRModel.LLM(model=..., gpu_memory_utilization=..., max_new_tokens=...)
    Qwen3ASRModel.from_pretrained(model, dtype=..., device_map=..., max_new_tokens=...)
    model.transcribe(audio=(np.ndarray, 16000), language="Korean") -> [ASRTranscription(.text)]
    state = model.init_streaming_state(language="Korean", chunk_size_sec=..., unfixed_chunk_num=..., unfixed_token_num=...)
    model.streaming_transcribe(pcm16k, state); state.text        (decodes when a full chunk is buffered)
    model.finish_streaming_transcribe(state); state.text         (flushes the tail)

`--mock` replaces the model with a dummy system so the whole pipeline (simulator, VAD,
records, evaluate.py) can be exercised on a CPU; its outputs are not results.
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

LANGUAGE = "Korean"


# --------------------------------------------------------------------------- #
# Qwen3-ASR adapters
# --------------------------------------------------------------------------- #
def load_qwen(cfg: dict):
    from qwen_asr import Qwen3ASRModel  # pip install -U qwen-asr[vllm]

    backend = cfg.get("backend", "vllm")
    if backend == "vllm":
        return Qwen3ASRModel.LLM(model=cfg["model"], gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.8),
                                 max_new_tokens=cfg.get("max_new_tokens", 256))
    import torch
    return Qwen3ASRModel.from_pretrained(cfg["model"], dtype=torch.bfloat16, device_map="cuda:0",
                                         max_new_tokens=cfg.get("max_new_tokens", 256))


class QwenOffline:
    def __init__(self, cfg: dict):
        self.model = load_qwen(cfg)

    def transcribe(self, audio, sample_rate: int) -> str:
        res = self.model.transcribe(audio=(audio, sample_rate), language=LANGUAGE)
        return res[0].text if res else ""


class QwenStreaming:
    def __init__(self, cfg: dict):
        if cfg.get("backend", "vllm") != "vllm":
            raise ValueError("qwen-asr streaming requires backend='vllm'")
        self.model = load_qwen(cfg)
        st = cfg.get("streaming", {})
        self.kw = dict(language=LANGUAGE, chunk_size_sec=st.get("chunk_sec", 1.0),
                       unfixed_chunk_num=st.get("unfixed_chunk_num", 2),
                       unfixed_token_num=st.get("unfixed_token_num", 5))
        self.state = None

    def reset(self) -> None:
        self.state = self.model.init_streaming_state(**self.kw)

    def push(self, chunk, sample_rate: int) -> Optional[str]:
        assert sample_rate == 16000, "qwen-asr streaming expects 16 kHz PCM"
        self.model.streaming_transcribe(chunk, self.state)
        return self.state.text or None

    def finalize(self) -> str:
        self.model.finish_streaming_transcribe(self.state)
        return self.state.text or ""


class Mock:
    """CPU stand-in: offline returns the reference-length placeholder; streaming grows per chunk."""

    def __init__(self, cfg: dict):
        import time
        self.delay = 0.02
        self.n = 0
        self._sleep = time.sleep

    def transcribe(self, audio, sample_rate: int) -> str:
        self._sleep(self.delay)
        return "목 결과 " * max(1, int(len(audio) / sample_rate))

    def reset(self) -> None:
        self.n = 0

    def push(self, chunk, sample_rate: int) -> Optional[str]:
        self._sleep(self.delay)
        self.n += 1
        return ("목 " * self.n).strip() if self.n > 1 else None

    def finalize(self) -> str:
        return ("목 " * self.n).strip() + " 끝"


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
    ap.add_argument("--repeats", type=int, default=None, help="override latency_config repeats")
    ap.add_argument("--mock", action="store_true", help="dummy system (pipeline test, no GPU)")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    lcfg = load_config(args.latency_config) if args.latency_config else load_config()
    if args.repeats:
        lcfg["repeats"] = args.repeats
    sr = lcfg["sample_rate"]
    streaming = cfg["mode"] == "streaming"
    if args.mock:
        system = Mock(cfg)
    else:
        system = QwenStreaming(cfg) if streaming else QwenOffline(cfg)

    utts = list(read_jsonl(args.manifest))
    if args.conditions:
        keep = set(args.conditions.split(","))
        utts = [u for u in utts if u["condition"] in keep]
    if args.limit:
        utts = utts[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    env = environment_info()
    env.update({"config": cfg, "mock": args.mock, "repeats": lcfg["repeats"],
                "n_utterances": len(utts)})
    with open(os.path.join(args.out_dir, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)

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
            lat_f.flush()
        pred_f.write(json.dumps({"id": u["id"], "text": text}, ensure_ascii=False) + "\n")
        pred_f.flush()
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(utts)}", flush=True)

    pred_f.close()
    if lat_f:
        lat_f.close()
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
