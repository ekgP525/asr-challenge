from evaluation.normalizer import normalize, normalize_nospace, int_to_korean, convert_numbers


def test_int_to_korean():
    assert int_to_korean(0) == "영"
    assert int_to_korean(1) == "일"
    assert int_to_korean(10) == "십"
    assert int_to_korean(11) == "십일"
    assert int_to_korean(20) == "이십"
    assert int_to_korean(100) == "백"
    assert int_to_korean(2026) == "이천이십육"
    assert int_to_korean(10000) == "만"
    assert int_to_korean(15000) == "만오천"
    assert int_to_korean(120000) == "십이만"
    assert int_to_korean(100000000) == "일억"


def test_convert_numbers_variants():
    assert convert_numbers("1학년") == "일학년"
    assert convert_numbers("1,000원") == "천원"
    assert convert_numbers("3.5점") == "삼점오점"
    assert convert_numbers("031-219") == "공삼일-이백십구"
    assert convert_numbers("2026년 2학기") == "이천이십육년 이학기"


def test_normalize_full():
    assert normalize("휴학 신청은, 언제까지  해야 하나요?") == "휴학 신청은 언제까지 해야 하나요"
    assert normalize("gpa 3.0 이상") == "GPA 삼점영 이상"
    assert normalize("등록금 50% 감면") == "등록금 오십퍼센트 감면"
    assert normalize_nospace("재 수강 !!") == "재수강"


def test_decimal_point_spelling_unified():
    assert normalize_nospace("삼 쩜 삼 제곱미터") == normalize_nospace("3.3 제곱미터")


def test_normalize_idempotent():
    s = "복학 신청은 2월 말까지 입니다."
    assert normalize(normalize(s)) == normalize(s)
