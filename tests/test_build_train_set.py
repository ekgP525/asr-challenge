"""Training-set mixing: level normalization, SNR, padding, clean floor (no downloads)."""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scripts"))
from build_train_set import SR, TARGET_LEVEL, mix_one  # noqa: E402
from p56 import active_speech_level, rms_db  # noqa: E402


def _speech(seconds=3.0, amp=0.05):
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * seconds)) / SR
    env = (np.sin(2 * np.pi * 3 * t) > 0).astype(np.float32)          # on/off bursts like syllables
    return (amp * np.sin(2 * np.pi * 180 * t) * env).astype(np.float32) + rng.standard_normal(len(t)).astype(np.float32) * 1e-4


def test_noisy_mix_hits_snr_and_pads():
    sp = _speech()
    noise = (np.random.default_rng(1).standard_normal(SR * 8) * 0.1).astype(np.float32)
    out, info = mix_one(sp, noise, 5.0, 0.4, 0.8, random.Random(3), 7)
    assert abs(len(out) / SR - (3.0 + 0.4 + 0.8)) < 1e-3
    assert info["speech_start"] == 0.4 and abs(info["speech_end"] - 3.4) < 1e-3
    assert abs(info["speech_level_dbov"] - TARGET_LEVEL) < 1.0
    # noise level in the leading pad should sit ~5 dB below the speech level
    lead = out[: int(0.4 * SR)]
    assert abs((info["speech_level_dbov"] - rms_db(lead)) - 5.0) < 1.5
    assert info["condition"] == "noisy" and float(np.abs(out).max()) <= 0.99


def test_clean_mix_has_floor_not_zeros():
    sp = _speech()
    out, info = mix_one(sp, None, None, 0.2, 0.5, random.Random(3), 7)
    lead = out[: int(0.2 * SR)]
    assert info["condition"] == "clean"
    assert -63 < rms_db(lead) < -57
    assert float(np.abs(lead).max()) > 0
