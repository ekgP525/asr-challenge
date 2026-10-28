from evaluation.latency import endpoint_error, final_latency, percentile, revision_count, summarize


def rec(**kw):
    base = dict(id="u", audio_duration=6.0, speech_start=1.0, speech_end=5.0, vad_end=5.35,
                t_final=5.8, t_first_partial=None, processing_time=0.4, partials=[], text="")
    base.update(kw)
    return base


def test_percentile_nearest_rank():
    xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert percentile(xs, 90) in (0.9, 1.0)
    assert percentile([0.42], 90) == 0.42


def test_final_latency_and_endpoint():
    r = rec()
    assert abs(final_latency(r) - 0.8) < 1e-9
    assert abs(endpoint_error(r) - 0.35) < 1e-9
    # VAD never fired -> audio end counts as detection
    assert abs(endpoint_error(rec(vad_end=None)) - 1.0) < 1e-9
    # early cut -> negative
    assert endpoint_error(rec(vad_end=4.2)) < 0


def test_revision_count_prefix_rule():
    parts = [{"t": 1, "text": "오늘"}, {"t": 2, "text": "오늘 날씨"}, {"t": 3, "text": "오늘 날씨가"},
             {"t": 4, "text": "오늘 낯씨가 좋다"}, {"t": 5, "text": "오늘 낯씨가 좋다."}]
    rev, upd = revision_count(parts)
    assert (rev, upd) == (1, 4)
    assert revision_count([]) == (0, 0)
    assert revision_count([{"t": 1, "text": "a"}]) == (0, 0)
    # whitespace-only differences are not revisions
    assert revision_count([{"t": 1, "text": "a  b"}, {"t": 2, "text": "a b c"}]) == (0, 1)


def test_summarize_offline():
    s = summarize([rec(), rec(vad_end=4.5, t_final=5.2)], streaming=False)
    assert s["n"] == 2
    assert s["early_cut"] == 50.0
    assert s["revision"] is None and s["first_partial"] is None
    assert s["rtfx"] == 15.0
    assert s["p90"] == 0.8


def test_summarize_streaming():
    parts = [{"t": 1.5, "text": "안녕"}, {"t": 2.0, "text": "안녕하"}, {"t": 2.5, "text": "안녕히"}]
    s = summarize([rec(t_first_partial=1.5, partials=parts)], streaming=True)
    assert s["revision"] == 50.0
    assert s["first_partial"] == 0.5
