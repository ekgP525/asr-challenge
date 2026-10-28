import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scripts"))
from mix_noise import tile  # noqa: E402
from p56 import active_speech_level, rms_db  # noqa: E402

SR = 16000


def test_p56_on_stationary_noise_equals_rms():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(SR * 3) * 0.1).astype(np.float32)
    lvl, act = active_speech_level(x, SR)
    assert abs(lvl - rms_db(x)) < 1.0
    assert act > 0.9


def test_p56_ignores_silence():
    rng = np.random.default_rng(1)
    burst = (rng.standard_normal(SR) * 0.1).astype(np.float32)
    x = np.concatenate([np.zeros(SR * 2, np.float32), burst, np.zeros(SR * 2, np.float32)])
    lvl, act = active_speech_level(x, SR)
    # active level ~ burst rms (-20 dB), not the long-term rms (-27 dB)
    assert abs(lvl - rms_db(burst)) < 1.5
    assert 0.15 < act < 0.35


def test_p56_on_real_speech():
    import glob
    sf = pytest.importorskip("soundfile")
    wavs = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "data", "eval", "audio", "src", "*.wav")))
    if not wavs:
        pytest.skip("no source wav")
    x, sr = sf.read(wavs[0], dtype="float32")
    lvl, act = active_speech_level(x, sr)
    assert -40 < lvl < -8          # read speech, normal recording level
    assert 0.4 < act < 0.98        # mostly active with pauses
    assert lvl > rms_db(x)         # active level above long-term rms
    assert lvl - rms_db(x) < 6


def test_tile_wraps_and_keeps_length():
    noise = np.arange(1000, dtype=np.float32) / 1000
    out = tile(noise, 2500, 300, 10)
    assert len(out) == 2500
    assert np.allclose(out[:10], noise[300:310])
    # offset near the end of the clip and offset beyond the clip length both work
    assert len(tile(noise, 2500, 995, 10)) == 2500
    assert np.allclose(tile(noise, 100, 1300, 10), tile(noise, 100, 300, 10))
    # seams are crossfaded: no jump larger than the ramp step
    seam = tile(noise, 3000, 0, 10)
    assert np.abs(np.diff(seam)).max() < 0.2


def test_snr_gain_formula():
    rng = np.random.default_rng(2)
    speech = (rng.standard_normal(SR * 2) * 0.2).astype(np.float32)
    noise = (rng.standard_normal(SR * 2) * 0.05).astype(np.float32)
    s_level, _ = active_speech_level(speech, SR)
    n_level = rms_db(noise)
    for snr in (20, 0):
        g = 10 ** ((s_level - n_level - snr) / 20)
        got = rms_db(speech) - rms_db(g * noise)
        assert abs(got - snr) < 1.0
