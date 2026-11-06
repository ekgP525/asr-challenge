"""A system may bring its own endpointer; the simulator must use it and charge its cost."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from evaluation.stream_sim import (  # noqa: E402
    build_endpointer, describe_endpointer, load_config, run_offline, run_streaming,
)


class Echo:
    def __init__(self):
        self.n = 0
    def transcribe(self, audio, sr):
        return "오프라인"
    def reset(self):
        self.n = 0
    def push(self, chunk, sr):
        self.n += 1
        return "부분" * self.n
    def finalize(self):
        return "부분" * self.n + " 끝"


class FixedEndpointer:
    """Fires start at `start_s`, end at `end_s`; records whether it saw the system."""
    def __init__(self, cfg, params=None, system=None):
        self.sr = cfg["sample_rate"]
        self.start = int(params["start_s"] * self.sr)
        self.end = int(params["end_s"] * self.sr)
        self.system = system
        self.reset()
    def reset(self):
        self.pos = 0
        self.fired = set()
    def feed(self, chunk):
        a, self.pos = self.pos, self.pos + len(chunk)
        out = {"start": None, "end_detect": None}
        if a <= self.start < self.pos and "s" not in self.fired:
            self.fired.add("s"); out["start"] = self.start
        if a < self.end <= self.pos and "e" not in self.fired:
            self.fired.add("e"); out["end_detect"] = self.end
        return out


def synthetic(sr, seconds=6.0):
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(int(sr * seconds)) * 1e-3).astype("float32")
    t = np.arange(int(sr * 3.0)) / sr
    a[sr * 1: sr * 4] += (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")   # "speech" 1-4 s
    return a


def test_custom_endpointer_drives_the_simulator():
    cfg = dict(load_config()); sr = cfg["sample_rate"]
    spec = {"type": "custom", "module": "tests.test_endpointer_plugin", "class": "FixedEndpointer",
            "params": {"start_s": 1.0, "end_s": 4.5}}
    sys_ = Echo()
    ep = build_endpointer(cfg, spec, sys_)
    assert type(ep).__name__ == "FixedEndpointer" and ep.system is sys_   # pytest imports the test module twice
    utt = {"id": "t", "speech_start": 1.0, "speech_end": 4.0}
    audio = synthetic(sr)
    r = run_streaming(audio, sr, utt, sys_, cfg, ep)
    assert r.vad_start == pytest.approx(1.0) and r.vad_end == pytest.approx(4.5)
    assert r.t_final >= 4.5                                 # clock moved to the detection position
    assert r.partials[-1]["text"].endswith("끝")
    r2 = run_offline(audio, sr, utt, sys_, cfg, ep)
    assert r2.vad_end == pytest.approx(4.5) and r2.text == "오프라인"
    assert describe_endpointer(spec) == "custom:tests.test_endpointer_plugin.FixedEndpointer"
    assert describe_endpointer(None) == "default"
    assert describe_endpointer({"type": "silero", "vad": {"min_silence_duration_ms": 300}}) == \
        "silero(min_silence_duration_ms=300)"


def test_energy_endpointer_example():
    cfg = dict(load_config()); sr = cfg["sample_rate"]
    spec = json.load(open(os.path.join(ROOT, "participants", "S0_energy_example.json")))["endpointer"]
    ep = build_endpointer(cfg, spec, None)
    utt = {"id": "t", "speech_start": 1.0, "speech_end": 4.0}
    r = run_streaming(synthetic(sr), sr, utt, Echo(), cfg, ep)
    assert r.vad_start is not None and abs(r.vad_start - 1.0) < 0.2
    assert r.vad_end is not None and 4.0 <= r.vad_end < 4.8      # end + ~400 ms silence
    assert r.processing_time > 0                                  # endpointer cost is charged


def test_silero_override_builds():
    pytest.importorskip("silero_vad")
    cfg = load_config()
    ep = build_endpointer(cfg, {"type": "silero", "vad": {"min_silence_duration_ms": 300}})
    assert ep.it.min_silence_samples == int(cfg["sample_rate"] * 300 / 1000)


def test_bad_spec():
    with pytest.raises(ValueError):
        build_endpointer(load_config(), {"type": "magic"})


def test_mock_runner_with_custom_endpointer(tmp_path):
    """End-to-end on CPU: runner (--mock) -> evaluate -> leaderboard, endpointer label carried through."""
    pytest.importorskip("soundfile")
    import soundfile as sf
    cfg = load_config(); sr = cfg["sample_rate"]
    (tmp_path / "audio").mkdir()
    rows = []
    for cond in ["clean", "snr20", "snr10", "snr5", "snr0"]:
        sf.write(str(tmp_path / "audio" / f"u_{cond}.wav"), synthetic(sr), sr)
        rows.append({"id": f"u_{cond}", "audio": f"audio/u_{cond}.wav", "condition": cond, "source": "zeroth",
                     "text": "테스트 문장", "duration": 6.0, "speech_start": 1.0, "speech_end": 4.0})
    man = tmp_path / "m.jsonl"
    man.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    conf = os.path.join(ROOT, "participants", "S0_energy_example.json")
    out = tmp_path / "runs"
    subprocess.run([sys.executable, os.path.join(ROOT, "baseline", "run_qwen3_asr.py"), "--config", conf,
                    "--manifest", str(man), "--audio-root", str(tmp_path), "--out-dir", str(out), "--mock"],
                   check=True, cwd=ROOT)
    env = json.load(open(out / "environment.json"))
    assert env["endpointer"] == "custom:participants.energy_endpointer.EnergyEndpointer"
    res = tmp_path / "S0.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "evaluate.py"), "--config", conf,
                    "--manifest", str(man), "--predictions", str(out / "predictions.jsonl"),
                    "--latency", str(out / "latency.jsonl"), "--out", str(res)], check=True, cwd=ROOT)
    r = json.load(open(res))
    assert r["endpointer"].startswith("custom:") and r["cond"]["snr0"]["endpoint"] is not None
    rd = tmp_path / "results"; rd.mkdir(); (rd / "S0.json").write_text(res.read_text())
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "make_leaderboard.py"), "--results-dir", str(rd),
                    "--out-md", str(rd / "lb.md"), "--out-json", str(rd / "lb.json"), "--no-plot"], check=True, cwd=ROOT)
    md = (rd / "lb.md").read_text(encoding="utf-8")
    assert "| Endpointer |" in md and "custom:participants.energy_endpointer.EnergyEndpointer" in md
