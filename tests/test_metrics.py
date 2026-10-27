import json
import os
import subprocess
import sys

from evaluation.metrics import (
    CERAccumulator, TermRecallAccumulator, cer, find_terms, levenshtein, term_recall,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_levenshtein():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("휴학", "복학") == 1
    assert levenshtein("kitten", "sitting") == 3


def test_cer_corpus_level():
    refs = ["휴학 신청", "복학 신청"]  # 4 + 4 chars after space removal
    hyps = ["휴학 신청", "복학 심청"]  # 0 + 1 edit
    assert abs(cer(refs, hyps) - 12.5) < 1e-9
    assert cer(["휴학"], [""]) == 100.0


def test_cer_ignores_spacing_and_punct():
    assert cer(["재수강은 언제 하나요?"], ["재수강은언제하나요"]) == 0.0


def test_term_recall():
    hyps = ["휴학 신청은 언제까지", "재수강 신청 방법"]
    terms = [["휴학"], ["재수강", "복학"]]
    # 휴학 hit, 재수강 hit, 복학 miss -> 2/3
    assert abs(term_recall(hyps, terms) - 200 / 3) < 1e-6


def test_term_recall_records_misses():
    acc = TermRecallAccumulator()
    acc.add("u1", "휴악 신청", ["휴학"])
    assert acc.total == 1 and acc.hits == 0
    assert acc.missed == [{"id": "u1", "term": "휴학"}]


def test_term_recall_substring_semantics():
    # Substring matching ignores spacing: '수강신청' is found inside '재수강 신청'.
    assert term_recall(["재수강 신청 방법"], [["수강신청"]]) == 100.0


def test_find_terms():
    terms = ["휴학", "복학", "학사경고"]
    assert find_terms("학사 경고를 받으면 휴학해야 하나요", terms) == ["휴학", "학사경고"]


def test_evaluate_end_to_end(tmp_path):
    manifest = [
        {"id": "a1", "text": "휴학 신청은 언제까지 해야 하나요", "terms": ["휴학"], "condition": "clean"},
        {"id": "a2", "text": "재수강 신청 방법", "terms": ["재수강"], "condition": "snr20"},
        {"id": "a3", "text": "학사경고 기준", "terms": ["학사경고"], "condition": "snr0"},
    ]
    preds = [
        {"id": "a1", "text": "휴학 신청은 언제까지 해야 하나요?"},
        {"id": "a2", "text": "재수강 신청 방법"},
        {"id": "a3", "text": "학사 경고 기준"},
    ]
    lat = [
        {"id": "a1", "audio_duration": 3.0, "final_latency": 0.4, "first_partial_latency": None, "processing_time": 0.3},
        {"id": "a2", "audio_duration": 2.0, "final_latency": 0.6, "first_partial_latency": None, "processing_time": 0.2},
        {"id": "a3", "audio_duration": 2.0, "final_latency": 0.9, "first_partial_latency": None, "processing_time": 0.5},
    ]
    def dump(name, rows):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        return str(p)
    out = tmp_path / "B1.json"
    cmd = [sys.executable, os.path.join(ROOT, "evaluation", "evaluate.py"),
           "--manifest", dump("m.jsonl", manifest), "--predictions", dump("p.jsonl", preds),
           "--latency", dump("l.jsonl", lat), "--config", os.path.join(ROOT, "baseline", "configs", "B1.json"),
           "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=ROOT)
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["system_id"] == "B1"
    assert r["track_a"]["clean"]["cer"] == 0.0
    assert r["track_a"]["noisy_avg"]["term_recall"] == 100.0
    assert r["track_a"]["noisy_avg"]["conditions"] == ["snr20", "snr0"]
    assert r["track_b"]["final_latency_p90"] == 0.9
    assert r["track_b"]["rtfx"] == 7.0
    assert r["meets_latency_constraint"] is True

    # leaderboard over that single result
    rd = tmp_path / "results"; rd.mkdir(); (rd / "B1.json").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "make_leaderboard.py"),
                    "--results-dir", str(rd), "--out-md", str(rd / "lb.md"), "--out-json", str(rd / "lb.json"),
                    "--no-plot"], check=True, cwd=ROOT)
    lb = json.loads((rd / "lb.json").read_text(encoding="utf-8"))
    assert lb["main"][0]["system_id"] == "B1" and lb["main"][0]["rank"] == 1
