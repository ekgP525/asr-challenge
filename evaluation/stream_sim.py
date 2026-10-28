"""Real-time streaming simulator + shared Silero VAD endpointer (Track B front end).

Protocol (identical for every system):

* Audio is fed in fixed-size chunks (default 0.5 s) at real-time pace.
* Silero VAD with the parameters in `latency_config.json` runs on every chunk
  for every system, so end-of-speech detection is charged equally.
* Ground-truth `speech_start` / `speech_end` come from the manifest (VAD on the
  clean audio before noise mixing). The evaluation audio carries 1 s of noise
  before and after the utterance, so the endpointer has to find the end
  inside noise, as it would in the field.
* Offline systems: when the endpointer fires, everything streamed so far is
  sent to `transcribe()` in one call. If it fires before the true end, the
  tail of the utterance is lost (early cut) - that is measured, not hidden.
* Streaming systems: each chunk is pushed with `push()`; every change of the
  partial transcript is recorded with its wall-clock time. After the endpointer
  fires, `finalize()` returns the final text.
* If the endpointer never fires, the end of the audio counts as the detection.
* Warm-up utterances are discarded; each utterance is measured `repeats` times
  and the run with the median final latency is kept.

All times in an `UttRecord` are in *stream seconds* (0 = first sample fed),
so they are directly comparable with manifest timestamps.

Heavy dependencies (numpy, torch, silero-vad) are imported lazily.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
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
        """Feed one chunk; return the current partial transcript (or None)."""
    def finalize(self) -> str: ...


@dataclass
class UttRecord:
    id: str
    audio_duration: float
    speech_start: float
    speech_end: float
    vad_end: Optional[float]          # stream time when the endpointer fired (None = never)
    t_final: float                    # stream time when the final text was available
    t_first_partial: Optional[float]  # stream time of the first non-empty partial
    processing_time: float            # wall-clock spent inside the system
    partials: List[dict]              # [{"t": stream_time, "text": ...}] every change, final included
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# VAD endpointer
# --------------------------------------------------------------------------- #
class SileroEndpointer:
    """Streaming end-of-speech detector using Silero VAD's VADIterator."""

    def __init__(self, cfg: dict):
        from silero_vad import VADIterator, load_silero_vad

        self.sr = cfg["sample_rate"]
        v = cfg["vad"]
        self.window = v["window_size_samples"]
        self.model = load_silero_vad()
        self.it = VADIterator(self.model, threshold=v["threshold"], sampling_rate=self.sr,
                              min_silence_duration_ms=v["min_silence_duration_ms"],
                              speech_pad_ms=v["speech_pad_ms"])
        self.reset()

    def reset(self):
        self.it.reset_states()
        self._buf = None
        self.speech_started = False

    def feed(self, chunk) -> Optional[int]:
        """Feed a float32 chunk. Returns the sample offset *within the chunk* at which
        speech end was detected, or None."""
        import numpy as np
        import torch

        consumed_before = 0 if self._buf is None else len(self._buf)
        self._buf = chunk if self._buf is None else np.concatenate([self._buf, chunk])
        pos = 0
        fired_at = None
        while len(self._buf) - pos >= self.window:
            win = self._buf[pos: pos + self.window]
            pos += self.window
            ev = self.it(torch.from_numpy(np.ascontiguousarray(win)), return_seconds=False)
            if ev and "start" in ev:
                self.speech_started = True
            if ev and "end" in ev and fired_at is None:
                fired_at = pos - consumed_before  # offset inside this chunk
        self._buf = self._buf[pos:]
        return fired_at


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
def iter_chunks(audio, sr: int, chunk_ms: int):
    n = int(sr * chunk_ms / 1000)
    for i in range(0, len(audio), n):
        yield audio[i: i + n]


def _pace(t0: float, stream_pos_sec: float):
    delay = t0 + stream_pos_sec - time.perf_counter()
    if delay > 0:
        time.sleep(delay)


def run_offline(audio, sr: int, utt: dict, system: OfflineSystem, cfg: dict,
                vad: SileroEndpointer) -> UttRecord:
    import numpy as np

    vad.reset()
    duration = len(audio) / sr
    t0 = time.perf_counter()
    fed = 0
    vad_end = None
    for chunk in iter_chunks(audio, sr, cfg["chunk_ms"]):
        fed += len(chunk)
        _pace(t0, fed / sr)
        off = vad.feed(chunk)
        if off is not None:
            vad_end = (fed - len(chunk) + off) / sr
            break
    segment = np.asarray(audio[:fed])
    t_inf0 = time.perf_counter()
    text = system.transcribe(segment, sr)
    t_final = time.perf_counter()
    return UttRecord(
        id=utt["id"], audio_duration=duration,
        speech_start=utt["speech_start"], speech_end=utt["speech_end"],
        vad_end=vad_end, t_final=t_final - t0, t_first_partial=None,
        processing_time=t_final - t_inf0,
        partials=[{"t": t_final - t0, "text": text}], text=text,
    )


def run_streaming(audio, sr: int, utt: dict, system: StreamingSystem, cfg: dict,
                  vad: SileroEndpointer) -> UttRecord:
    vad.reset()
    system.reset()
    duration = len(audio) / sr
    t0 = time.perf_counter()
    fed = 0
    vad_end = None
    busy = 0.0
    partials: List[dict] = []
    last = ""
    for chunk in iter_chunks(audio, sr, cfg["chunk_ms"]):
        fed += len(chunk)
        _pace(t0, fed / sr)
        off = vad.feed(chunk)
        if off is not None:
            vad_end = (fed - len(chunk) + off) / sr
        tb = time.perf_counter()
        partial = system.push(chunk, sr)
        now = time.perf_counter()
        busy += now - tb
        if partial is not None and partial != last:
            partials.append({"t": now - t0, "text": partial})
            last = partial
        if vad_end is not None:
            break
    tb = time.perf_counter()
    text = system.finalize()
    t_final = time.perf_counter()
    busy += t_final - tb
    if text != last:
        partials.append({"t": t_final - t0, "text": text})
    first = next((p["t"] for p in partials if p["text"].strip()), None)
    return UttRecord(
        id=utt["id"], audio_duration=duration,
        speech_start=utt["speech_start"], speech_end=utt["speech_end"],
        vad_end=vad_end, t_final=t_final - t0, t_first_partial=first,
        processing_time=busy, partials=partials, text=text,
    )


def measure_median(run: Callable[[], UttRecord], repeats: int) -> UttRecord:
    """Run `repeats` times, keep the run whose final latency is the median."""
    runs = [run() for _ in range(repeats)]
    runs.sort(key=lambda r: r.t_final - r.speech_end)
    return runs[len(runs) // 2]
