"""Text normalization shared by CER and Term Recall.

Rules (fixed for the challenge; do not change without bumping the version):

1. Unicode NFC normalization.
2. Percent sign -> '퍼센트'.
3. Arabic numerals -> Sino-Korean reading (일, 이, 삼 ... 십, 백, 천, 만, 억).
   - "2026" -> "이천이십육", "10" -> "십", "1학년" -> "일학년"
   - Thousands separators are dropped first: "1,000" -> "천"
   - Decimals: "3.5" -> "삼점오", "3.0" -> "삼점영"
   - Numbers with a leading zero (phone numbers, codes) are read digit by digit,
     0 -> "공": "031" -> "공삼일"
   - Native Korean readings (한, 두, 세 ...) are NOT produced. Reference
     transcripts must therefore be written in Hangul with the reading the
     speaker actually used; the model output's digits are mapped to the
     Sino-Korean reading only. This is a known, documented limitation.
4. Latin letters -> upper case (English abbreviations unified: "gpa" -> "GPA").
5. All punctuation and symbols removed (Unicode categories P* and S*).
6. Whitespace collapsed to a single space, leading/trailing whitespace stripped.

`normalize()` keeps single spaces (for human-readable diffs);
`normalize_nospace()` additionally removes every space and is what CER and
Term Recall operate on.
"""
from __future__ import annotations

import re
import unicodedata

NORMALIZER_VERSION = "1.0"

_DIGITS = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
_DIGIT_ZERO = "공"
_SMALL_UNITS = ["", "십", "백", "천"]
_LARGE_UNITS = ["", "만", "억", "조"]

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _read_below_10000(n: int) -> str:
    """Read 0 < n < 10000 in Sino-Korean, omitting '일' before a unit (십, 백, 천)."""
    out = []
    for pos in range(3, -1, -1):
        d = (n // (10 ** pos)) % 10
        if d == 0:
            continue
        if pos == 0:
            out.append(_DIGITS[d])
        else:
            out.append(("" if d == 1 else _DIGITS[d]) + _SMALL_UNITS[pos])
    return "".join(out)


def int_to_korean(n: int) -> str:
    """Sino-Korean reading of a non-negative integer. 0 -> '영'."""
    if n < 0:
        raise ValueError("negative numbers are not supported")
    if n == 0:
        return "영"
    parts = []
    group_idx = 0
    while n > 0:
        n, grp = divmod(n, 10000)
        if grp:
            read = _read_below_10000(grp)
            # '일만' -> '만' (but keep '일억', '일조' as written? Korean reads 일억 as '일억')
            if group_idx == 1 and read == "일":
                read = ""
            parts.append(read + _LARGE_UNITS[group_idx])
        group_idx += 1
    return "".join(reversed(parts))


def number_token_to_korean(tok: str) -> str:
    """Convert one numeric token (digits, optional single decimal point)."""
    if "." in tok:
        whole, frac = tok.split(".", 1)
        whole_read = number_token_to_korean(whole) if whole else "영"
        frac_read = "".join("영" if c == "0" else _DIGITS[int(c)] for c in frac)
        return whole_read + "점" + frac_read
    if len(tok) > 1 and tok[0] == "0":
        return "".join(_DIGIT_ZERO if c == "0" else _DIGITS[int(c)] for c in tok)
    return int_to_korean(int(tok))


def convert_numbers(text: str) -> str:
    text = _THOUSANDS_RE.sub("", text)
    return _NUMBER_RE.sub(lambda m: number_token_to_korean(m.group(0)), text)


def _strip_punct(text: str) -> str:
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S"):
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def normalize(text: str) -> str:
    """Full normalization, single spaces kept."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("%", "퍼센트")
    text = convert_numbers(text)
    text = text.upper()
    text = _strip_punct(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_nospace(text: str) -> str:
    """Normalization followed by removal of all whitespace (CER / Term Recall input)."""
    return normalize(text).replace(" ", "")


if __name__ == "__main__":
    import sys

    for line in sys.stdin:
        print(normalize(line.rstrip("\n")))
