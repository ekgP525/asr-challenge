"""Extract the clean reference subset from a public Korean corpus.

Candidate sources (final choice TBD): zeroth-korean (CC BY 4.0),
FLEURS-ko (CC BY 4.0), Common Voice ko (CC0).

Output: 16 kHz mono wav files under data/eval/audio/clean_XXXX.wav and
manifest rows with condition="clean", source="<corpus>", speech_end from
Silero VAD on the clean audio.

TODO: implement once the corpus is chosen (see README "미확정 사항").
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=["zeroth", "fleurs", "commonvoice"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default="data/eval")
    ap.parse_args()
    raise SystemExit("not implemented yet")


if __name__ == "__main__":
    main()
