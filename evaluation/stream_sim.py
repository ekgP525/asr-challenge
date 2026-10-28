"""Streaming simulator (discrete-event real time) + shared Silero VAD endpointer.

Protocol (identical for every system):

* Audio is fed to the ASR in fixed-size chunks (default 0.5 s). Chunk i becomes
  available at stream time (i+1)*chunk; a virtual clock advances to that arrival
  time and then by the *measured wall-clock* cost of every VAD / model call.
* The ASR is VAD-gated, as in a deployed assistant: nothing is sent to the model
  until the endpointer reports speech start; the audio handed over begins
  `preroll_ms` before that start (streaming: the buffered audio from there to the
  end of the current chunk is the first push). Without gating, prefix-conditioned
  streaming recognizers lock in text hallucinated on the leading silence.
* The endpointer runs on 32 ms frames, so when it fires inside a chunk the
  detection is timed at that frame's arrival, not at the chunk boundary, and only
  the audio up to that frame is delivered to the ASR (offline: as the segment;
  streaming: as a last partial chunk before finalize()). A system that keeps
  up sees exactly real-time behaviour; one that falls behind accumulates queueing
  delay, as it would with a live microphone. No sleeping, so a run costs only the
  compute time, and results are deterministic given the measured costs.
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

All times in an `UttRecord` are stream seconds on the virtual clock (0 = first
sample fed), directly comparable with manifest timestamps.

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
    vad_start: Optional[float]        # speech start reported by the VAD (stream s, None = never)
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
        self.pos = 0            # samples consumed by the VAD (absolute stream position)
        self.speech_started = False

    def feed(self, chunk) -> dict:
        """Feed a float32 chunk. Returns {"start": abs_sample | None, "end_detect": abs_sample | None}:
        `start` is the speech start reported by the VAD (already moved back by speech_pad),
        `end_detect` is the absolute position of the frame at which the end was detected."""
        import numpy as np
        import torch

        self._buf = chunk if self._buf is None else np.concatenate([self._buf, chunk])
        out = {"start": None, "end_detect": None}
        i = 0
        while len(self._buf) - i >= self.window:
            win = self._buf[i: i + self.window]
            i += self.window
            self.pos += self.window
            ev = self.it(torch.from_numpy(np.ascontiguousarray(win)), return_seconds=False)
            if ev and "start" in ev and not self.speech_started:
                self.speech_started = True
                out["start"] = int(ev["start"])
            if ev and "end" in ev and out["end_detect"] is None:
                out["end_detect"] = self.pos
        self._buf = self._buf[i:]
        return out


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
def iter_chunks(audio, sr: int, chunk_ms: int):
    n = int(sr * chunk_ms / 1000)
    for i in range(0, len(audio), n):
        yield audio[i: i + n]


class _Clock:
    """Virtual stream clock: t = max(t, arrival) then t += measured compute."""

    def __init__(self):
        self.t = 0.0
        self.busy = 0.0

    def arrive(self, stream_pos: float):
        if stream_pos > self.t:
            self.t = stream_pos

    def spend(self, t_wall0: float) -> float:
        dt = time.perf_counter() - t_wall0
        self.t += dt
        self.busy += dt
        return dt


def run_offline(audio, sr: int, utt: dict, system: OfflineSystem, cfg: dict,
                vad: SileroEndpointer) -> UttRecord:
    import numpy as np

    vad.reset()
    clk = _Clock()
    duration = len(audio) / sr
    preroll = int(cfg.get("preroll_ms", 0) * sr / 1000)
    fed = 0
    start_sample = None
    end_sample = len(audio)
    vad_end = None
    for chunk in iter_chunks(audio, sr, cfg["chunk_ms"]):
        fed += len(chunk)
        w0 = time.perf_counter()
        ev = vad.feed(chunk)
        if ev["start"] is not None:
            start_sample = ev["start"]
        if ev["end_detect"] is not None:
            end_sample = ev["end_detect"]
            vad_end = end_sample / sr
            clk.arrive(vad_end)        # the frame that triggered the endpoint has just arrived
            clk.spend(w0)
            break
        clk.arrive(fed / sr)
        clk.spend(w0)
    if start_sample is None:           # VAD never saw speech: nothing is sent to the model
        return UttRecord(id=utt["id"], audio_duration=duration, speech_start=utt["speech_start"],
                         speech_end=utt["speech_end"], vad_start=None, vad_end=vad_end, t_final=clk.t,
                         t_first_partial=None, processing_time=0.0, partials=[], text="")
    seg0 = max(0, start_sample - preroll)
    segment = np.asarray(audio[seg0:end_sample])
    w0 = time.perf_counter()
    text = system.transcribe(segment, sr)
    infer = clk.spend(w0)
    return UttRecord(
        id=utt["id"], audio_duration=duration,
        speech_start=utt["speech_start"], speech_end=utt["speech_end"],
        vad_start=start_sample / sr, vad_end=vad_end, t_final=clk.t, t_first_partial=None,
        processing_time=infer,
        partials=[{"t": clk.t, "text": text}], text=text,
    )


def run_streaming(audio, sr: int, utt: dict, system: StreamingSystem, cfg: dict,
                  vad: SileroEndpointer) -> UttRecord:
    vad.reset()
    system.reset()
    clk = _Clock()
    duration = len(audio) / sr
    preroll = int(cfg.get("preroll_ms", 0) * sr / 1000)
    fed = 0
    start_sample = None
    vad_end = None
    partials: List[dict] = []
    last = ""

    def emit(partial):
        nonlocal last
        if partial is not None and partial != last:
            partials.append({"t": clk.t, "text": partial})
            last = partial

    def push(a, b):
        nonlocal last
        if b <= a:
            return
        w0 = time.perf_counter()
        emit(system.push(audio[a:b], sr))
        clk.spend(w0)

    for chunk in iter_chunks(audio, sr, cfg["chunk_ms"]):
        c0 = fed
        fed += len(chunk)
        w0 = time.perf_counter()
        ev = vad.feed(chunk)
        gate_open_at = None
        if ev["start"] is not None and start_sample is None:
            start_sample = ev["start"]
            gate_open_at = max(0, start_sample - preroll)   # first push starts here (buffered audio)
        if ev["end_detect"] is not None:
            vad_end = ev["end_detect"] / sr
            clk.arrive(vad_end)
            clk.spend(w0)
            if start_sample is not None:
                push(gate_open_at if gate_open_at is not None else c0, ev["end_detect"])
            break
        clk.arrive(fed / sr)
        clk.spend(w0)
        if start_sample is not None:
            push(gate_open_at if gate_open_at is not None else c0, fed)
    if start_sample is None:
        return UttRecord(id=utt["id"], audio_duration=duration, speech_start=utt["speech_start"],
                         speech_end=utt["speech_end"], vad_start=None, vad_end=vad_end, t_final=clk.t,
                         t_first_partial=None, processing_time=0.0, partials=[], text="")
    w0 = time.perf_counter()
    text = system.finalize()
    clk.spend(w0)
    emit(text)
    first = next((p["t"] for p in partials if p["text"].strip()), None)
    return UttRecord(
        id=utt["id"], audio_duration=duration,
        speech_start=utt["speech_start"], speech_end=utt["speech_end"],
        vad_start=start_sample / sr, vad_end=vad_end, t_final=clk.t, t_first_partial=first,
        processing_time=clk.busy, partials=partials, text=text,
    )


def measure_median(run: Callable[[], UttRecord], repeats: int) -> UttRecord:
    """Run `repeats` times, keep the run whose final latency is the median."""
    runs = [run() for _ in range(repeats)]
    runs.sort(key=lambda r: r.t_final - r.speech_end)
    return runs[len(runs) // 2]
