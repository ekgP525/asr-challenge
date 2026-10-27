"""Synthesize the campus-domain sentences (data/sentences/*.txt) with TTS.

Constraint: avoid Qwen-family TTS so the Qwen3-ASR baselines get no
same-family advantage. Output 16 kHz mono wav + manifest rows with
source="tts", domain, terms (filled with evaluation.metrics.find_terms).

TODO: implement once the TTS engine is chosen.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sentences", default="data/sentences")
    ap.add_argument("--terms", default="data/terms.txt")
    ap.add_argument("--out-dir", default="data/eval")
    ap.parse_args()
    raise SystemExit("not implemented yet")


if __name__ == "__main__":
    main()
