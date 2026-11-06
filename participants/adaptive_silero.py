"""Adaptive Silero endpointer (participant system S1).

Three changes to the default endpointer, each motivated by the baseline analysis:

1. Start detection unchanged (prob >= start_threshold, same speech_pad): the ASR gate opens exactly as in
   the baselines, so partial transcripts are identical and offline replay of recorded partials is exact.
2. End detection uses its own, higher thresholds: a window is *speech* (resets the silence timer) when
   prob >= end_speech_threshold and *silence* (counts toward the timer) when prob < end_silence_threshold.
   Babble / crowd noise at 0 dB keeps Silero around 0.5-0.8, which the default (0.5 / 0.35) treats as
   speech for seconds after the speaker stops; 0.8 / 0.65 cuts that tail (0 dB end-delay p90 1.04 -> 0.72 s
   in the endpointer-only sweep) at the cost of a few more early cuts. The higher thresholds apply only
   once speech has been confirmed at end_speech_threshold; before that the default hysteresis is kept so
   a weak onset is not immediately counted as silence.
3. Text-aware silence duration (optional, `use_text`): the endpointer reads the ASR's latest partial
   (`system.text`). Korean read sentences end with a sentence-final ending (다/요/까/죠/...) in >90% of the
   references, so when the partial ends with one, the timer is `min_silence_ms`; when it does not, the
   sentence is probably mid-pause (the early cuts in the baselines happen at in-sentence pauses > 500 ms)
   and the timer is extended to `min_silence_incomplete_ms`. Partials lag the audio (Qwen3-ASR decodes
   per 1.0 s chunk), so an incomplete-looking partial at a true end costs extra delay; the GPU run uses a
   0.5 s decode chunk to shorten that lag, and the offline replay (1.0 s partials) is the pessimistic case.

Config example:
    "endpointer": {"type": "custom", "module": "participants.adaptive_silero", "class": "AdaptiveSileroEndpointer",
                   "params": {"start_threshold": 0.5, "end_speech_threshold": 0.8, "end_silence_threshold": 0.65,
                              "min_silence_ms": 500, "min_silence_incomplete_ms": 800, "use_text": true}}
"""
from __future__ import annotations

import re

FINAL_ENDINGS = ("다", "요", "까", "죠", "네", "군", "라", "자", "오", "냐", "는가", "련", "지")
_TRAIL = re.compile(r"[\s\.\?\!,。…\"'\)\]]+$")


def looks_complete(text: str) -> bool:
    """True when a (partial) transcript ends with a Korean sentence-final ending."""
    t = _TRAIL.sub("", text or "")
    return bool(t) and t.endswith(FINAL_ENDINGS)


class AdaptiveSileroEndpointer:
    def __init__(self, cfg: dict, params: dict | None = None, system=None):
        from silero_vad import load_silero_vad

        p = params or {}
        v = cfg["vad"]
        self.sr = cfg["sample_rate"]
        self.window = v["window_size_samples"]
        self.start_thr = float(p.get("start_threshold", v["threshold"]))
        self.end_speech_thr = float(p.get("end_speech_threshold", 0.8))
        self.end_silence_thr = float(p.get("end_silence_threshold", 0.65))
        self.min_silence = self.sr * float(p.get("min_silence_ms", v["min_silence_duration_ms"])) / 1000
        self.min_silence_inc = self.sr * float(p.get("min_silence_incomplete_ms", 800)) / 1000
        self.use_text = bool(p.get("use_text", True))
        self.pad = self.sr * float(p.get("speech_pad_ms", v["speech_pad_ms"])) / 1000
        self.model = load_silero_vad()
        self.system = system
        self.reset()

    def reset(self) -> None:
        self.model.reset_states()
        self._buf = None
        self.pos = 0
        self.triggered = False
        self.confirmed = False      # speech seen at >= end_speech_threshold since the start event
        self.temp_end = 0
        self.done = False

    def _required_silence(self) -> float:
        if not self.use_text or self.system is None:
            return self.min_silence
        text = getattr(self.system, "text", None)
        if text is None:
            return self.min_silence
        return self.min_silence if looks_complete(text) else self.min_silence_inc

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
            if not self.triggered:
                if prob >= self.start_thr:
                    self.triggered = True
                    out["start"] = int(max(0, self.pos - self.pad - self.window))
                continue
            # until speech is confirmed at the higher threshold, keep the default hysteresis so a weak
            # onset (prob between start_threshold and end_speech_threshold) is not cut as silence
            if not self.confirmed and prob >= self.end_speech_thr:
                self.confirmed = True
            speech_thr = self.end_speech_thr if self.confirmed else self.start_thr
            silence_thr = self.end_silence_thr if self.confirmed else self.start_thr - 0.15
            if prob >= speech_thr:
                self.temp_end = 0
            elif prob < silence_thr:
                if not self.temp_end:
                    self.temp_end = self.pos
                elif self.pos - self.temp_end >= self._required_silence():
                    self.done = True
                    out["end_detect"] = self.pos
        self._buf = self._buf[i:]
        return out
