import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dump(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(path)


def utt(i, cond, text):
    return {"id": f"u{i}_{cond}", "text": text, "condition": cond, "source": "zeroth",
            "speech_start": 1.0, "speech_end": 4.0, "duration": 5.0}


def lat(uid, vad_end, t_final, partials=None):
    return {"id": uid, "audio_duration": 5.0, "speech_start": 1.0, "speech_end": 4.0, "vad_end": vad_end,
            "t_final": t_final, "t_first_partial": 1.6 if partials else None, "processing_time": 0.5,
            "partials": partials or [], "text": ""}


def run(args):
    subprocess.run([sys.executable] + args, check=True, cwd=ROOT)


def test_evaluate_and_leaderboard(tmp_path):
    conds = ["clean", "snr20", "snr10", "snr5", "snr0"]
    manifest = [utt(i, c, "휴학 신청은 언제까지") for c in conds for i in range(2)]
    # B1 offline: perfect clean, 1 error at snr0; latency fine everywhere
    preds1 = [{"id": u["id"], "text": "휴학 신청은 언제까지" if u["condition"] != "snr0" else "휴악 신청은 언제까지"}
              for u in manifest]
    lat1 = [lat(u["id"], 4.3, 4.6) for u in manifest]
    # B2 streaming: p90 over limit at snr0 -> outside ranking
    preds2 = [{"id": u["id"], "text": "휴학 신청은 언제까지"} for u in manifest]
    lat2 = [lat(u["id"], 4.3, 4.6 if u["condition"] != "snr0" else 5.3,
                partials=[{"t": 1.6, "text": "휴"}, {"t": 2.2, "text": "휴학"}]) for u in manifest]
    rd = tmp_path / "results"; rd.mkdir()
    m = dump(tmp_path / "m.jsonl", manifest)
    for sid, p, l in (("B1", preds1, lat1), ("B2", preds2, lat2)):
        run([os.path.join(ROOT, "evaluation", "evaluate.py"), "--manifest", m,
             "--predictions", dump(tmp_path / f"p{sid}.jsonl", p), "--latency", dump(tmp_path / f"l{sid}.jsonl", l),
             "--config", os.path.join(ROOT, "baseline", "configs", f"{sid}.json"), "--out", str(rd / f"{sid}.json")])
    r1 = json.loads((rd / "B1.json").read_text(encoding="utf-8"))
    assert set(r1["cond"]) == set(conds)
    assert r1["cond"]["clean"]["cer"] == 0.0 and r1["cond"]["snr0"]["cer"] > 0
    assert r1["cond"]["snr0"]["revision"] is None
    assert abs(r1["cond"]["snr0"]["endpoint"] - 0.3) < 1e-6
    assert r1["noisy_avg"]["p90"] == 0.6 and r1["meets_latency_constraint"] is True
    assert r1["rtfx"] == 10.0
    assert r1["by_source"]["zeroth"]["snr0"] == r1["cond"]["snr0"]["cer"]
    r2 = json.loads((rd / "B2.json").read_text(encoding="utf-8"))
    assert r2["meets_latency_constraint"] is False
    assert r2["cond"]["snr0"]["revision"] == 0.0 and r2["cond"]["snr0"]["first_partial"] == 0.6

    run([os.path.join(ROOT, "evaluation", "make_leaderboard.py"), "--results-dir", str(rd),
         "--out-md", str(rd / "lb.md"), "--out-json", str(rd / "lb.json"), "--no-plot"])
    lb = json.loads((rd / "lb.json").read_text(encoding="utf-8"))
    assert [x["system_id"] for x in lb["main"]] == ["B1"]
    assert [x["system_id"] for x in lb["other"]] == ["B2"]
    assert lb["main"][0]["rank"] == 1
