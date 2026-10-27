"""Track B: streaming simulator and latency measurement.

Protocol (identical for every system):

* Audio is fed in fixed-size chunks (default 0.5 s) at real-time pace.
* Silero VAD with the parameters in `latency_config.json` runs on every chunk
  for every system, so end-of-speech detection delay is charged equally.
* Speech end is the ground-truth `speech_end` timestamp from the manifest, not
  the VAD output. Final Latency = wall-clock time of final text - wall-clock time
  at which `speech_end` was reached in the stream.
* Offline systems: once VAD reports end of speech the whole utterance is sent
  to `transcribe()` in one call. Final Latency = VAD detection delay + inference.
* Streaming systems: each chunk is pushed with `push()`; the first non-empty
  partial result gives First-partial Latency (measured from `speech_start`).
  After the last chunk `finalize()` returns the final text.
* Warm-up utterances are discarded; every utterance is measured `repeats`
  times and the median is kept. RTFx = audio duration / processing time.

Heavy dependencies (numpy, torch, silero-vad) are imported lazily so the
accuracy metrics stay importable on a CPU-only machine.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Protocol

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "latency_config.json")


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# System interfaces
# --------------------------------------------------------------------------- #
class OfflineSystem(Protocol):
    def transcribe(self, audio, sample_rate: int) -> str: ...


class StreamingSystem(Protocol):
    def reset(self) -> None: ...
    def push(self, chunk, sample_rate: int) -> Optional[str]:
        """Feed one chunk; return the current partial transcript or None."""
    def finalize(self) -> str: ...


@dataclass
class UttLatency:
    id: str
    audio_duration: float
    final_latency: float
    first_partial_latency: Optional[float]
    processing_time: float
    vad_end_delay: Optional[float]
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# VAD
# --------------------------------------------------------------------------- #
class SileroEndpointer:
    """Streaming end-of-speech detector using Silero VAD's VADIterator."""

    def __init__(self, cfg: dict):
        import torch  # noqa: F401
        from silero_vad import VADIterator, load_silero_vad

        self.sr = cfg["sample_rate"]
        v = cfg["vad"]
        self.window = v["window_size_samples"]
        self.model = load_silero_vad()
        self.it = VADIterator(
            self.model,
            threshold=v["threshold"],
            sampling_rate=self.sr,
            min_silence_duration_ms=v["min_silence_duration_ms"],
            speech_pad_ms=v["speech_pad_ms"],
        )
        self.reset()

    def reset(self):
        self.it.reset_states()
        self._buf = None
        self.speech_started = False
        self.speech_ended = False

    def feed(self, chunk) -> bool:
        """Feed a float32 numpy chunk; return True on the chunk where speech end fires."""
        import numpy as np
        import torch

        self._buf = chunk if self._buf is None else np.concatenate([self._buf, chunk])
        fired = False
        while len(self._buf) >= self.window:
            win, self._buf = self._buf[: self.window], self._buf[self.window :]
            ev = self.it(torch.from_numpy(win), return_seconds=False)
            if ev and "start" in ev:
                self.speech_started = True
            if ev and "end" in ev:
                self.speech_ended = True
                fired = True
        return fired


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
def iter_chunks(audio, sr: int, chunk_ms: int):
    n = int(sr * chunk_ms / 1000)
    for i in range(0, len(audio), n):
        yield audio[i : i + n]


def _pace(t0: float, stream_pos_sec: float):
    """Sleep until wall-clock has caught up with the stream position."""
    delay = t0 + stream_pos_sec - time.perf_counter()
    if delay > 0:
        time.sleep(delay)


def run_offline(
    audio, sr: int, utt_id: str, speech_end: float, system: OfflineSystem,
    cfg: dict, vad: Optional[SileroEndpointer] = None,
) -> UttLatency:
    import numpy as np

    vad = vad or SileroEndpointer(cfg)
    vad.reset()
    chunk_ms = cfg["chunk_ms"]
    duration = len(audio) / sr
    t0 = time.perf_counter()
    pos = 0.0
    t_detect = None
    for chunk in iter_chunks(audio, sr, chunk_ms):
        pos += len(chunk) / sr
        _pace(t0, pos)
        if vad.feed(chunk):
            t_detect = time.perf_counter()
            break
    if t_detect is None:  # VAD never fired (short tail): end of audio counts as detection
        t_detect = time.perf_counter()
    t_inf0 = time.perf_counter()
    text = system.transcribe(np.asarray(audio), sr)
    t_final = time.perf_counter()
    t_speech_end_wall = t0 + speech_end
    return UttLatency(
        id=utt_id,
        audio_duration=duration,
        final_latency=t_final - t_speech_end_wall,
        first_partial_latency=None,
        processing_time=t_final - t_inf0,
        vad_end_delay=t_detect - t_speech_end_wall,
        text=text,
    )


def run_streaming(
    audio, sr: int, utt_id: str, speech_start: float, speech_end: float,
    system: StreamingSystem, cfg: dict, vad: Optional[SileroEndpointer] = None,
) -> UttLatency:
    vad = vad or SileroEndpointer(cfg)
    vad.reset()
    system.reset()
    chunk_ms = cfg["chunk_ms"]
    duration = len(audio) / sr
    t0 = time.perf_counter()
    pos = 0.0
    t_first_partial = None
    t_detect = None
    busy = 0.0
    for chunk in iter_chunks(audio, sr, chunk_ms):
        pos += len(chunk) / sr
        _pace(t0, pos)
        if vad.feed(chunk) and t_detect is None:
            t_detect = time.perf_counter()
        tb = time.perf_counter()
        partial = system.push(chunk, sr)
        busy += time.perf_counter() - tb
        if partial and t_first_partial is None:
            t_first_partial = time.perf_counter()
        if t_detect is not None:
            break  # speech ended: stop feeding, finalize
    tb = time.perf_counter()
    text = system.finalize()
    t_final = time.perf_counter()
    busy += t_final - tb
    t_speech_end_wall = t0 + speech_end
    return UttLatency(
        id=utt_id,
        audio_duration=duration,
        final_latency=t_final - t_speech_end_wall,
        first_partial_latency=(t_first_partial - (t0 + speech_start)) if t_first_partial else None,
        processing_time=busy,
        vad_end_delay=(t_detect - t_speech_end_wall) if t_detect else None,
        text=text,
    )


def measure_median(run: Callable[[], UttLatency], repeats: int) -> UttLatency:
    """Run `repeats` times, keep the run with the median final latency."""
    runs = [run() for _ in range(repeats)]
    runs.sort(key=lambda r: r.final_latency)
    return runs[len(runs) // 2]


# --------------------------------------------------------------------------- #
# Aggregation (pure Python; also used by evaluate.py)
# --------------------------------------------------------------------------- #
def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


def summarize(records: List[dict]) -> dict:
    finals = [r["final_latency"] for r in records if r.get("final_latency") is not None]
    firsts = [r["first_partial_latency"] for r in records if r.get("first_partial_latency") is not None]
    audio = sum(r.get("audio_duration", 0.0) for r in records)
    proc = sum(r.get("processing_time", 0.0) for r in records)
    return {
        "final_latency_p50": round(percentile(finals, 50), 4) if finals else None,
        "final_latency_p90": round(percentile(finals, 90), 4) if finals else None,
        "first_partial_latency": round(statistics.median(firsts), 4) if firsts else None,
        "rtfx": round(audio / proc, 2) if proc > 0 else None,
        "n_utts": len(records),
    }
