"""Participant endpointer S1: sentence-final detection, text-aware silence, weak-onset guard; tool smoke tests."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from participants.adaptive_silero import looks_complete  # noqa: E402


def test_looks_complete():
    assert looks_complete("염소 사육은 이란에서 시작한 것으로 보입니다.")
    assert looks_complete("그렇게 했다")
    assert looks_complete("가능할까요? ")
    assert not looks_complete("염소사육은 대략 일만 년 전에 이란의.")
    assert not looks_complete("염소사육은 대략 일만년")
    assert not looks_complete("")


class Talker:
    """Streaming stub whose partial text is set by the test."""
    text = ""
    def reset(self): ...
    def push(self, chunk, sr): return None
    def finalize(self): return ""


WAVS = sorted(__import__("glob").glob(os.path.join(ROOT, "data", "eval", "audio", "src", "*.wav")))


def _tone_with_pause(sr, pause_s):
    """Synthetic signal for tool smoke tests (Silero does not treat it as speech): noise, tone, gap, tone, noise."""
    rng = np.random.default_rng(1)
    noise = lambda s: (rng.standard_normal(int(sr * s)) * 1e-3).astype("float32")
    tone = lambda s: (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(sr * s)) / sr)).astype("float32")
    return np.concatenate([noise(1), tone(2), noise(pause_s), tone(1.5), noise(1.5)])


def _speech_with_pause(sr, pause_s):
    """Real speech with an inserted mid-utterance pause: 1 s noise, first 2 s of a clean utterance,
    `pause_s` of noise, the next 1.5 s of the same utterance, 1.5 s noise. Returns (audio, true_end)."""
    import soundfile as sf
    a, fsr = sf.read(WAVS[0], dtype="float32")
    assert fsr == sr
    rng = np.random.default_rng(1)
    noise = lambda s: (rng.standard_normal(int(sr * s)) * 1e-3).astype("float32")
    part1, part2 = a[: int(sr * 2.0)], a[int(sr * 2.0): int(sr * 3.5)]
    audio = np.concatenate([noise(1), part1, noise(pause_s), part2, noise(1.5)]).astype("float32")
    return audio, 1.0 + 2.0 + pause_s + 1.5


@pytest.mark.skipif(not WAVS, reason="no clean wav extracted yet")
def test_text_aware_silence_extends_only_when_incomplete():
    pytest.importorskip("silero_vad"); pytest.importorskip("soundfile")
    from evaluation.stream_sim import build_endpointer, load_config, run_streaming
    cfg = load_config(); sr = cfg["sample_rate"]
    spec = {"type": "custom", "module": "participants.adaptive_silero", "class": "AdaptiveSileroEndpointer",
            "params": {"min_silence_ms": 500, "min_silence_incomplete_ms": 1000, "use_text": True}}
    audio, true_end = _speech_with_pause(sr, 0.7)   # 700 ms pause: cut by a 500 ms timer, survived by 1000 ms
    utt = {"id": "t", "speech_start": 1.0, "speech_end": true_end}
    sysA = Talker(); sysA.text = "아직 끝나지 않은 문장의"     # incomplete -> 900 ms timer -> survives the pause
    rA = run_streaming(audio, sr, utt, sysA, cfg, build_endpointer(cfg, spec, sysA))
    sysB = Talker(); sysB.text = "문장이 끝났습니다."         # complete -> 500 ms timer -> cut at the pause
    rB = run_streaming(audio, sr, utt, sysB, cfg, build_endpointer(cfg, spec, sysB))
    assert rA.vad_end is not None and rA.vad_end >= utt["speech_end"] - 0.3     # survived the pause
    assert rB.vad_end is not None and rB.vad_end < utt["speech_end"] - 1.0       # cut at the pause


def test_endpointer_tools_smoke(tmp_path):
    """endpointer_eval (with --replay) and endpointer_whatif run end to end on a tiny synthetic set."""
    pytest.importorskip("silero_vad"); pytest.importorskip("soundfile")
    import soundfile as sf
    from evaluation.stream_sim import load_config
    sr = load_config()["sample_rate"]
    (tmp_path / "audio").mkdir()
    rows, lat, preds = [], [], []
    for cond in ["clean", "snr20", "snr10", "snr5", "snr0"]:
        uid = f"u_{cond}"
        sf.write(str(tmp_path / "audio" / f"{uid}.wav"), _tone_with_pause(sr, 0.0), sr)
        rows.append({"id": uid, "audio": f"audio/{uid}.wav", "condition": cond, "source": "zeroth", "text": "테스트 문장이다",
                     "duration": 6.0, "speech_start": 1.0, "speech_end": 4.5, "noise": {"type": "crowd"}})
        lat.append({"id": uid, "audio_duration": 6.0, "speech_start": 1.0, "speech_end": 4.5, "vad_start": 1.0,
                    "vad_end": 5.0, "t_final": 5.1, "t_first_partial": 2.0, "processing_time": 0.3,
                    "partials": [{"t": 2.009, "text": "테스트"}, {"t": 4.009, "text": "테스트 문장이다."}], "text": "테스트 문장이다."})
        preds.append({"id": uid, "text": "테스트 문장이다."})
    w = lambda name, items: (tmp_path / name).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items), encoding="utf-8")
    w("m.jsonl", rows); run = tmp_path / "run"; run.mkdir()
    (run / "latency.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lat), encoding="utf-8")
    (run / "predictions.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in preds), encoding="utf-8")
    ep_out = tmp_path / "ep.jsonl"
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "endpointer_eval.py"), "--spec", os.path.join(ROOT, "participants", "S1.json"),
                    "--manifest", str(tmp_path / "m.jsonl"), "--audio-root", str(tmp_path), "--replay", str(run / "latency.jsonl"),
                    "--out", str(ep_out), "--quiet"], check=True, cwd=ROOT)
    assert len(ep_out.read_text(encoding="utf-8").strip().splitlines()) == 5
    out = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "endpointer_whatif.py"), "--ep", str(ep_out),
                          "--run", str(run), "--manifest", str(tmp_path / "m.jsonl"), "--json", str(tmp_path / "w.json")],
                         check=True, cwd=ROOT, capture_output=True, text=True)
    assert "noisy CER est" in out.stdout
    assert "cond" in json.loads((tmp_path / "w.json").read_text(encoding="utf-8"))
