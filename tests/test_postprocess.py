from evaluation.postprocess import collapse_repeats


def test_collapse_repeats_truncates_loops():
    loop = "교육정책을 추진하는 것을 목표로 하는 "
    out = collapse_repeats("교육부가 서울 주요 대학을 중심으로 한 " + loop * 8 + "끝")
    assert out == ("교육부가 서울 주요 대학을 중심으로 한 " + loop).rstrip()


def test_collapse_repeats_is_inert_on_normal_text():
    for t in ["염소사육은 대략 일만 년 전에 이란의 자그로스 산맥에서 시작한 것으로 보입니다.",
              "네 네 네 알겠습니다", "하하하하하하하하", ""]:
        assert collapse_repeats(t) == t
