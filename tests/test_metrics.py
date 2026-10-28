from evaluation.metrics import CERAccumulator, cer, levenshtein, utterance_cer


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


def test_utterance_cer():
    assert utterance_cer("가나다라", "가나다") == 25.0
    assert utterance_cer("", "x") == 0.0
