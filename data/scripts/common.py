"""Shared helpers for the dataset-building scripts (clean subset, TTS, augmentation)."""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from evaluation.stream_sim import load_config as load_latency_config  # noqa: E402

SR = load_latency_config()["sample_rate"]  # 16000


def to_mono_float32(audio, sr: int, target_sr: int = SR) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(audio)


def write_wav(path: str, audio: np.ndarray, sr: int = SR) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


def read_wav(path: str, target_sr: int = SR) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    return to_mono_float32(audio, sr, target_sr)


class SpeechBounds:
    """Silero VAD based speech_start / speech_end (seconds) for a whole utterance.

    Parameters are fixed here (threshold 0.5, min silence 300 ms, pad 30 ms) and
    independent of the endpointer config, so the reference bounds in the published
    manifests do not change when the latency protocol is tuned.
    """

    THRESHOLD = 0.5
    MIN_SILENCE_MS = 300
    SPEECH_PAD_MS = 30

    def __init__(self):
        from silero_vad import get_speech_timestamps, load_silero_vad

        self.model = load_silero_vad()
        self._get = get_speech_timestamps
        self.kw = dict(threshold=self.THRESHOLD, min_silence_duration_ms=self.MIN_SILENCE_MS,
                       speech_pad_ms=self.SPEECH_PAD_MS, sampling_rate=SR, return_seconds=True)

    def __call__(self, audio: np.ndarray) -> Optional[tuple]:
        import torch
        ts = self._get(torch.from_numpy(audio), self.model, **self.kw)
        if not ts:
            return None
        return float(ts[0]["start"]), float(ts[-1]["end"])


def write_jsonl(path: str, rows) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
