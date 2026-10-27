from evaluation.latency import percentile, summarize


def test_percentile_nearest_rank():
    xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert percentile(xs, 50) == 0.6 or percentile(xs, 50) == 0.5
    assert percentile(xs, 90) == 1.0 or percentile(xs, 90) == 0.9
    assert percentile([0.42], 90) == 0.42


def test_summarize_streaming():
    recs = [
        {"final_latency": 0.5, "first_partial_latency": 0.3, "audio_duration": 2.0, "processing_time": 0.5},
        {"final_latency": 0.7, "first_partial_latency": 0.4, "audio_duration": 2.0, "processing_time": 0.5},
    ]
    s = summarize(recs)
    assert s["n_utts"] == 2
    assert s["first_partial_latency"] == 0.35
    assert s["rtfx"] == 4.0
