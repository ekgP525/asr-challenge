"""Energy-gated Silero endpointer (participant system S1 candidate).

Observation from the baselines: at 0 dB the *median* end delay is still ~0.58 s (min silence 500 ms), but a
long tail (crowd / babble noise, p90 2.4 s) keeps Silero's speech probability high after the speaker has
stopped, so the default endpointer lingers. Babble, however, does not raise the signal energy above the
noise floor the way the near-field speaker does (~+3 dB at 0 dB SNR). So:

* Start detection is exactly Silero's VADIterator logic (threshold, speech_pad) -> the ASR gate opens at
  the same moment as the baseline, partial transcripts are unaffected.
* End detection: a 32 ms window counts as *silence* when Silero says non-speech (prob < threshold - 0.15,
  the VADIterator hysteresis) OR its energy is within `margin_db` of the tracked noise floor. The silence
  timer is reset only by a window that is speech by both criteria. End fires after `min_silence_ms` of
  silence, timed at the window that completed it (same convention as the default endpointer).
* Noise floor: running mean (dB) of the windows before speech start - the evaluation audio, like a real
  device, has leading noise - then tracked slowly on non-speech windows (`floor_alpha`).

Config example:
    "endpointer": {"type": "custom", "module": "participants.gated_silero", "class": "GatedSileroEndpointer",
                   "params": {"threshold": 0.5, "min_silence_ms": 500, "speech_pad_ms": 30,
                              "margin_db": 3.0, "floor_alpha": 0.02}}
"""
from __future__ import annotations

import math


class GatedSileroEndpointer:
    def __init__(self, cfg: dict, params: dict | None = None, system=None):
        from silero_vad import load_silero_vad

        p = params or {}
        v = cfg["vad"]
        self.sr = cfg["sample_rate"]
        self.window = v["window_size_samples"]
        self.threshold = float(p.get("threshold", v["threshold"]))
        self.min_silence = self.sr * float(p.get("min_silence_ms", v["min_silence_duration_ms"])) / 1000
        self.pad = self.sr * float(p.get("speech_pad_ms", v["speech_pad_ms"])) / 1000
        self.margin = float(p.get("margin_db", 3.0))
        self.floor_alpha = float(p.get("floor_alpha", 0.02))
        self.model = load_silero_vad()
        self.system = system
        self.reset()

    def reset(self) -> None:
        self.model.reset_states()
        self._buf = None
        self.pos = 0
        self.triggered = False
        self.temp_end = 0
        self.floor = None
        self.n_floor = 0
        self.done = False

    @staticmethod
    def _db(x) -> float:
        return 10 * math.log10(float((x.astype("float64") ** 2).mean()) + 1e-10)

    def feed(self, chunk) -> dict:
        import numpy as np
        import torch

        self._buf = chunk if self._buf is None else np.concatenate([self._buf, chunk])
        out = {"start": None, "end_detect": None}
        i = 0
        while len(self._buf) - i >= self.window:
            win = np.ascontiguousarray(self._buf[i: i + self.window])
            i += self.window
            self.pos += self.window
            if self.done:
                continue
            with torch.no_grad():
                prob = self.model(torch.from_numpy(win), self.sr).item()
            db = self._db(win)
            silero_speech = prob >= self.threshold
            silero_silence = prob < self.threshold - 0.15

            if not self.triggered:
                # noise floor from the leading non-speech audio
                if not silero_speech:
                    self.floor = db if self.floor is None else self.floor + 0.2 * (db - self.floor)
                    self.n_floor += 1
                if silero_speech:
                    self.triggered = True
                    if self.floor is None:
                        self.floor = db - 10.0          # no leading noise seen: assume well below speech
                    out["start"] = int(max(0, self.pos - self.pad - self.window))
                continue

            above = self.floor is None or db >= self.floor + self.margin
            if silero_silence or not above:
                if not above and silero_silence:
                    self.floor += self.floor_alpha * (db - self.floor)
                if not self.temp_end:
                    self.temp_end = self.pos
                elif self.pos - self.temp_end >= self.min_silence:
                    self.done = True
                    out["end_detect"] = self.pos
                    self.triggered = False
            elif silero_speech and above:
                self.temp_end = 0
        self._buf = self._buf[i:]
        return out
