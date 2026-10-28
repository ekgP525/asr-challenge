"""Integration check of the streaming simulator with the real Silero VAD.

Skipped when torch / silero-vad / a clean wav are not available (CPU-only CI is fine).
"""
import glob
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAVS = sorted(glob.glob(os.path.join(ROOT, "data", "eval", "audio", "clean", "*.wav")))


class Echo:
    """Offline + streaming dummy system: returns a text that grows with the audio seen."""
    def __init__(self):
        self.n = 0
    def transcribe(self, audio, sr):
        return "오프라인 결과"
    def reset(self):
        self.n = 0
    def push(self, chunk, sr):
        self.n += 1
        return "부분" * self.n
    def finalize(self):
        return "부분" * self.n + " 끝"


@pytest.mark.skipif(not WAVS, reason="no clean wav extracted yet")
def test_simulator_with_silero_vad():
    np = pytest.importorskip("numpy")
    pytest.importorskip("silero_vad")
    sf = pytest.importorskip("soundfile")
    from evaluation.stream_sim import SileroEndpointer, load_config, run_offline, run_streaming

    cfg = dict(load_config())
    cfg["chunk_ms"] = 500
    sr = cfg["sample_rate"]
    audio, file_sr = sf.read(WAVS[0], dtype="float32")
    assert file_sr == sr
    # pad 1 s of low noise before/after like the evaluation audio will have
    rng = np.random.default_rng(0)
    pad = (rng.standard_normal(sr) * 1e-3).astype("float32")
    audio = np.concatenate([pad, audio, pad]).astype("float32")
    # ground-truth bounds: quick VAD on the padded audio itself
    from silero_vad import get_speech_timestamps, load_silero_vad
    ts = get_speech_timestamps(__import__("torch").from_numpy(audio), load_silero_vad(),
                               sampling_rate=sr, return_seconds=True)
    utt = {"id": "t", "speech_start": ts[0]["start"], "speech_end": ts[-1]["end"]}
    # speed up: feed faster than real time by shrinking the pacing (monkeypatch)
    import evaluation.stream_sim as ss
    ss._pace = lambda t0, pos: None
    vad = SileroEndpointer(cfg)

    r = run_offline(audio, sr, utt, Echo(), cfg, vad)
    assert r.vad_end is not None and r.vad_end >= utt["speech_end"] - 0.1
    assert r.vad_end - utt["speech_end"] < 1.0           # detected inside the 1 s tail
    assert r.text == "오프라인 결과" and len(r.partials) == 1

    r2 = run_streaming(audio, sr, utt, Echo(), cfg, vad)
    assert r2.vad_end is not None
    assert r2.t_first_partial is not None
    assert r2.partials[-1]["text"].endswith("끝")
    assert len(r2.partials) >= 3
