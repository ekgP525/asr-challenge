# CampusVoice-RT: 캠퍼스 현장 한국어 질의 인식 챌린지

캠퍼스 현장(야외·원거리·잡음)에서 들어오는 한국어 학사·행정 질의를 **1초 안에** 받아쓰는 음성인식 챌린지.
2026-2학기 아주대학교 음성인식 수업 개인 프로젝트 (이다혜, 202422051).

- 챌린지 홈페이지: (GitHub Pages 링크 TBD)
- Primary metric: 잡음 조건(SNR 20/10/5/0 dB) 평균 **Term Recall**
- 제약: **p90 Final Latency ≤ 1.0 s** (NVIDIA L4, 동일 VAD 설정)

## Tracks

| Track | 지표 |
|---|---|
| A. 정확도 | CER (%) ↓, Term Recall (%) ↑ — Clean / SNR 20·10·5·0 dB |
| B. 실시간성 | Final Latency p50/p90 (s) ↓, First-partial Latency (s) ↓ (스트리밍만), RTFx ↑ |

순위: 잡음 평균 Term Recall → (동률) 잡음 평균 CER. p90 Final Latency가 1.0 s를 넘는 시스템은 순위 밖 별도 표기.

## 레포 구조

```
baseline/
  run_qwen3_asr.py        Qwen3-ASR 1.7B/0.6B, offline/streaming, context prompt
  configs/B1..B5.json     베이스라인 설정
data/
  eval/manifest.jsonl     평가셋 manifest (오디오는 HF Datasets)
  terms.txt               Term Recall 용어 사전
  sentences/              도메인 문장 원문 + 출처
  scripts/                클린 서브셋 / TTS / 잡음·잔향 합성
evaluation/
  normalizer.py           텍스트 정규화 (고정 규칙, 버전 관리)
  metrics.py              CER, Term Recall
  latency.py              스트리밍 시뮬레이터, Silero VAD, 지연 측정
  latency_config.json     VAD·청크·반복 횟수 등 측정 프로토콜
  evaluate.py             예측 → results/<system>.json
  make_leaderboard.py     results/ → 순위표(md/json) + 지연–Term Recall 산점도
results/                  시스템별 결과 JSON
notebooks/reproduce_colab.ipynb
tests/                    평가 코드 단위 테스트
```

## 평가 코드 사용법

```bash
# 1) 베이스라인 실행 (Colab L4)
python baseline/run_qwen3_asr.py --config baseline/configs/B1.json \
    --manifest data/eval/manifest.jsonl --audio-root data/eval --out-dir runs/B1

# 2) 채점 → results/B1.json
python evaluation/evaluate.py --config baseline/configs/B1.json \
    --manifest data/eval/manifest.jsonl \
    --predictions runs/B1/predictions.jsonl --latency runs/B1/latency.jsonl \
    --env runs/B1/environment.json --hardware "NVIDIA L4"

# 3) 리더보드 + 산점도
python evaluation/make_leaderboard.py
```

예측 파일 형식: jsonl, 한 줄에 `{"id": "<manifest id>", "text": "<모델 출력>"}`.
외부 시스템을 평가하려면 예측 파일만 이 형식으로 만들면 된다(Track A). Track B까지 포함하려면
`evaluation/latency.py`의 `OfflineSystem` / `StreamingSystem` 인터페이스를 구현해 같은 시뮬레이터로 측정한다.

### 정규화 규칙 (`evaluation/normalizer.py`, v1.0)

1. NFC 정규화, `%` → `퍼센트`
2. 아라비아 숫자 → 한자어 숫자 읽기 (`2026` → `이천이십육`, `1,000` → `천`, `3.5` → `삼점오`, 선행 0은 자리별 읽기 `031` → `공삼일`)
3. 영문자 대문자 통일
4. 문장부호·기호 제거, 연속 공백 1개로
5. CER·Term Recall은 공백까지 제거한 문자열로 계산

고유어 숫자(한, 두, 세)는 생성하지 않으므로 정답 transcript는 화자가 실제 읽은 한글 표기로 적는다.

### 지연 측정 프로토콜 (`evaluation/latency_config.json`)

- 청크 0.5 s, 실시간 속도로 공급. Silero VAD threshold 0.5, min silence 300 ms, speech pad 30 ms — 모든 시스템 동일.
- Final Latency = 정답 `speech_end` 시점(스트림 기준 wall-clock) → 최종 텍스트 출력.
- 오프라인: VAD 종료 감지 후 전체 발화 1회 추론. 스트리밍: 청크마다 `push()`, 첫 부분결과로 First-partial Latency.
- 워밍업 3발화 제외, 발화당 3회 반복 중 중앙값. RTFx = 오디오 길이 ÷ 처리 시간.

## 테스트

```bash
pip install pytest
python -m pytest
```

## 실행 환경

Colab, NVIDIA L4 고정. GPU·CUDA·torch/transformers/vLLM/qwen-asr 버전은 각 결과 JSON의 `environment`에 기록.

## 라이선스 / 출처

- 평가 코드·스크립트: MIT (TBD)
- 데이터: 공개 배포 가능한 소스만 사용 (zeroth-korean / FLEURS-ko / Common Voice ko, MUSAN, OpenSLR SLR28). AI Hub 자료는 재배포 금지라 미사용.
- 베이스라인: [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) ([1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B))
