# 참가 시스템 만들기

참가 시스템은 **ASR + 엔드포인터** 한 쌍입니다. 둘 다 바꿔도 되고 한쪽만 바꿔도 됩니다. 측정은 베이스라인과 같은
시뮬레이터(`evaluation/stream_sim.py`)로 하며, 엔드포인터의 연산 시간도 같은 가상 시계에 더해집니다.

## 1. ASR

`evaluation/stream_sim.py`의 인터페이스를 구현합니다.

- 오프라인: `transcribe(audio, sample_rate) -> str`. 엔드포인터가 종료를 검출한 시점까지의 오디오를 한 번에 받습니다.
- 스트리밍: `reset()`, `push(chunk, sample_rate) -> 부분결과 | None`, `finalize() -> str`. 0.5초 청크가 이어 들어오고,
  종료 검출 뒤 `finalize()`가 호출됩니다.

`baseline/run_qwen3_asr.py`의 `QwenOffline` / `QwenStreaming`이 예시입니다. 다른 모델을 쓰려면 같은 모양의 클래스를
만들고 러너에서 생성하도록 바꾸거나, 러너를 복사해 쓰면 됩니다.

## 2. 엔드포인터

기본값은 모든 베이스라인이 쓴 Silero VAD(threshold 0.5, min silence 500 ms)입니다. 설정 파일의 `endpointer` 키로 바꿉니다.

```jsonc
// 기본값 (생략 가능)
"endpointer": "default"

// Silero 파라미터만 조정
"endpointer": {"type": "silero", "vad": {"min_silence_duration_ms": 350, "threshold": 0.6}}

// 직접 구현
"endpointer": {"type": "custom", "module": "participants.energy_endpointer", "class": "EnergyEndpointer",
               "params": {"min_silence_ms": 400, "snr_db": 6.0}}
```

직접 구현하는 클래스는 `__init__(self, cfg, params, system)`, `reset()`, `feed(chunk) -> {"start", "end_detect"}`를
가집니다(`participants/energy_endpointer.py` 참고). `system`은 같이 돌아가는 ASR 객체라, 부분결과를 보고 종료를
판단하는 식의 결합도 가능합니다(같은 청크에서는 `feed()`가 `push()`보다 먼저 호출됩니다).

지켜야 할 것:

- `end_detect`는 **그 판단에 쓴 마지막 오디오의 위치**여야 합니다. 시뮬레이터가 시계를 그 위치로 옮기므로, 아직 도착하지 않은
  오디오를 보고 판단한 것처럼 꾸밀 수 없습니다.
- 정답 `speech_end`는 manifest에 있지만 엔드포인터가 읽어서는 안 됩니다. 시뮬레이터는 오디오만 넘깁니다.
- 종료 검출 오차·조기 절단율이 시스템별로 리더보드에 실리므로, 공격적인 종료 판정은 지연을 줄이는 대신 조기 절단과 CER로 드러납니다.

## 3. 실행·채점

```bash
python baseline/run_qwen3_asr.py --config participants/S1.json \
    --manifest data/eval/manifest.jsonl --audio-root data/eval --out-dir runs/S1
python evaluation/evaluate.py --config participants/S1.json --manifest data/eval/manifest.jsonl \
    --predictions runs/S1/predictions.jsonl --latency runs/S1/latency.jsonl \
    --env runs/S1/environment.json --hardware "NVIDIA L4"
python evaluation/make_leaderboard.py
```

`system_id`는 B1~B4와 겹치지 않게 짓습니다. 결과 JSON과 리더보드에 어떤 엔드포인터를 썼는지 기록됩니다.
GPU 없이 파이프라인만 확인하려면 `--mock --limit 20`을 붙입니다.
