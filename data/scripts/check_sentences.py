"""Validate data/sentences/*.tsv and report term coverage.

Checks
* every row is `sentence<TAB>url`, url starts with http
* no Arabic digits in sentences (reference transcripts must be in Hangul)
* no duplicate sentences across files
* each sentence contains at least one dictionary term (warn otherwise)
* term frequency (terms that never appear are listed so the dictionary stays honest)

Usage: python data/scripts/check_sentences.py [--sentences data/sentences] [--terms data/terms.txt]
"""
import argparse
import collections
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from evaluation.metrics import find_terms, load_terms  # noqa: E402
from evaluation.normalizer import normalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sentences", default=os.path.join(ROOT, "data", "sentences"))
    ap.add_argument("--terms", default=os.path.join(ROOT, "data", "terms.txt"))
    args = ap.parse_args()

    terms = load_terms(args.terms)
    errors, warnings = [], []
    seen = {}
    freq = collections.Counter()
    total = 0
    per_file = {}
    for path in sorted(glob.glob(os.path.join(args.sentences, "*.tsv"))):
        name = os.path.basename(path)
        n = 0
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 2 or not parts[1].startswith("http"):
                    errors.append(f"{name}:{ln}: expected 'sentence<TAB>url'")
                    continue
                s = parts[0].strip()
                if re.search(r"\d", s):
                    errors.append(f"{name}:{ln}: Arabic digit in sentence: {s}")
                if s in seen:
                    errors.append(f"{name}:{ln}: duplicate of {seen[s]}: {s}")
                seen[s] = f"{name}:{ln}"
                if normalize(s) != s:
                    warnings.append(f"{name}:{ln}: not normalization-stable: {s!r} -> {normalize(s)!r}")
                hits = find_terms(s, terms)
                if not hits:
                    warnings.append(f"{name}:{ln}: no dictionary term: {s}")
                freq.update(hits)
                n += 1
        per_file[name] = n
        total += n

    print("sentences per file:", per_file, "total:", total)
    print(f"terms in dictionary: {len(terms)}, terms used: {len(freq)}")
    unused = [t for t in terms if t not in freq]
    if unused:
        print(f"unused terms ({len(unused)}): " + ", ".join(unused))
    print("top terms:", freq.most_common(15))
    for w in warnings:
        print("WARN", w)
    for e in errors:
        print("ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
