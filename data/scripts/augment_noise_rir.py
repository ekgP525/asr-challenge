"""Create the noisy / reverberant conditions from the clean + domain audio.

For every clean utterance produce snr20 / snr10 / snr5 / snr0 versions:
    y = rir * x  (OpenSLR SLR28 room impulse response, random pick)
    y = y + noise * gain   (MUSAN noise/babble, gain set for the target SNR)
Random choices are driven by --seed (default 2026) so anyone can regenerate
the exact evaluation set. Manifest rows copy id/text/terms/speech_end from the
clean row with id suffix "_snr20" etc. and condition="snrNN".

TODO: implement (numpy + scipy.signal.fftconvolve + soundfile).
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/eval/manifest_clean.jsonl")
    ap.add_argument("--musan", required=True, help="path to MUSAN root")
    ap.add_argument("--rir", required=True, help="path to SLR28 RIRS_NOISES root")
    ap.add_argument("--snrs", default="20,10,5,0")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default="data/eval")
    ap.parse_args()
    raise SystemExit("not implemented yet")


if __name__ == "__main__":
    main()
