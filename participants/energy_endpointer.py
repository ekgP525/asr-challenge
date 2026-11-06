"""Example custom endpointer: adaptive energy-based end-of-speech detector (no neural VAD).

Shows the interface the simulator expects (evaluation/stream_sim.py `Endpointer`); replace the
logic with your own. Everything here runs on 32 ms frames so that the detection position is a
frame boundary, exactly like the default Silero endpointer.

    reset()            -> called before every utterance
    feed(chunk) -> {"start": abs_sample | None, "end_detect": abs_sample | None}

`start` is reported once (first speech frame, moved back by `pad_ms`); `end_detect` is the absolute
sample position at which the end was decided. The simulator advances the virtual clock to that
position and then by the measured wall-clock cost of feed(), so a slow endpointer pays for it.

Config (baseline/configs/*.json or your own):
    "endpointer": {"type": "custom", "module": "participants.energy_endpointer",
                   "class": "EnergyEndpointer",
                   "params": {"min_silence_ms": 400, "snr_db": 6.0, "pad_ms": 30}}
"""
from __future__ import annotations

import math


class EnergyEndpointer:
    def __init__(self, cfg: dict, params: dict | None = None, system=None):
        p = params or {}
        self.sr = cfg["sample_rate"]
        self.frame = int(self.sr * 0.032)                      # 32 ms frames, like Silero
        self.min_silence = int(p.get("min_silence_ms", 400) / 32)
        self.min_speech = int(p.get("min_speech_ms", 96) / 32)
        self.snr_db = float(p.get("snr_db", 6.0))              # speech must exceed the noise floor by this
        self.pad = int(self.sr * p.get("pad_ms", 30) / 1000)
        self.floor_alpha = float(p.get("floor_alpha", 0.05))   # noise-floor tracking rate on non-speech frames
        self.system = system                                   # unused here; available for ASR-informed logic
        self.reset()

    def reset(self) -> None:
        self._buf = None
        self.pos = 0
        self.floor = None           # running estimate of the noise-floor energy (dB)
        self.speech_frames = 0
        self.silence_frames = 0
        self.started = False
        self.done = False

    @staticmethod
    def _db(frame) -> float:
        e = float((frame.astype("float64") ** 2).mean())
        return 10 * math.log10(e + 1e-10)

    def feed(self, chunk) -> dict:
        import numpy as np

        self._buf = chunk if self._buf is None else np.concatenate([self._buf, chunk])
        out = {"start": None, "end_detect": None}
        i = 0
        while len(self._buf) - i >= self.frame:
            f = self._buf[i: i + self.frame]
            i += self.frame
            self.pos += self.frame
            if self.done:
                continue
            db = self._db(f)
            if self.floor is None:
                self.floor = db
            speech = db > self.floor + self.snr_db
            if not speech:
                self.floor += self.floor_alpha * (db - self.floor)
            if not self.started:
                self.speech_frames = self.speech_frames + 1 if speech else 0
                if self.speech_frames >= self.min_speech:
                    self.started = True
                    out["start"] = max(0, self.pos - self.speech_frames * self.frame - self.pad)
            else:
                self.silence_frames = 0 if speech else self.silence_frames + 1
                if self.silence_frames >= self.min_silence:
                    self.done = True
                    out["end_detect"] = self.pos
        self._buf = self._buf[i:]
        return out
