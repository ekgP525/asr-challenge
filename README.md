# CampusVoice-RT: 야외 잡음 환경 한국어 실시간 음성인식 챌린지

잡음이 커질수록 음성인식은 **정확도**와 **응답 지연**을 함께 잃는다. 이 챌린지는 공개 한국어 코퍼스에 야외 잡음을
SNR 20/10/5/0 dB로 합성한 평가셋에서, 실시간 스트리밍 조건으로 둘의 관계를 측정한다.
2026-2학기 아주대학교 음성인식 수업 개인 프로젝트 (이다혜, 202422051).

- 시나리오: 학생회관 앞 잔디밭의 마스코트 음성 비서. 주변 대화·바람·차량·행사 소리 속에서, 말을 마친 뒤 1초 안에 인식 완료.
- 챌린지 홈페이지: (GitHub Pages 링크 TBD)
- Primary metric: 잡음 4조건 평균 **CER**
- 제약: 잡음 4조건 **각각** p90 최종 지연 ≤ 1.0 s (NVIDIA L4, 동일 VAD 설정)

## Tracks

| Track | 조건 | 지표 |
|---|---|---|
| A. 정확도 | Clean / SNR 20·10·5·0 dB | CER (%) ↓ |
| B. 실시간성 | 동일 | 최종 지연 p50/p90 (s) ↓ · 종료 검출 오차 (s) ↓ · 조기 절단율 (%) ↓ · 부분결과 수정률 (%) ↓ (스트리밍) · 첫 부분결과 지연 (s) ↓ (스트리밍) · RTFx ↑ |

순위: 잡음 4조건 평균 CER → 동률 시 평균 p90. 한 조건이라도 p90 > 1.0 s면 순위 밖 별도 표기.

## 레포 구조

```
baseline/
  run_qwen3_asr.py        Qwen3-ASR 1.7B/0.6B × offline/streaming (B1~B4)
  configs/B1..B4.json
data/
  eval/                   manifest_clean_<source>.jsonl (클린 중간), manifest.jsonl (최종), clean_subset_ids_<source>.txt
  noise_list.csv          사용 잡음 파일 + 개별 라이선스 (TBD)
  LICENSES/               Zeroth, FLEURS, MUSAN, FSD50K 출처 표기 (TBD)
  scripts/
    select_subset.py      Zeroth·FLEURS test에서 발화 추출 + VAD 정답 구간
    filter_noise.py       FSD50K CC0·CC BY 필터, 범주 선택 (TBD)
    mix_noise.py          P.56 기준 SNR 4조건 합성, 앞뒤 1 s 잡음 패딩, 시드 고정 (TBD)
    common.py
evaluation/
  normalizer.py           텍스트 정규화 (고정 규칙, 버전 관리)
  metrics.py              CER
  stream_sim.py           실시간 공급 시뮬레이터 + Silero VAD 엔드포인터
  latency.py              최종 지연, 종료 검출 오차, 조기 절단율, 수정률, RTFx
  latency_config.json     VAD·청크·반복 횟수 등 측정 프로토콜
  evaluate.py             예측 + 지연 기록 → results/<system>.json
  make_leaderboard.py     results/ → 순위표(md/json) + Clean→0 dB 경로 그래프
results/
notebooks/reproduce_colab.ipynb
tests/
```

## 평가 코드 사용법

```bash
# 1) 베이스라인 실행 (Colab L4)
python baseline/run_qwen3_asr.py --config baseline/configs/B1.json \
    --manifest data/eval/manifest.jsonl --audio-root data/eval --out-dir runs/B1

# 2) 채점 → results/B1.json
python evaluation/evaluate.py --config baseline/configs/B1.json --manifest data/eval/manifest.jsonl \
    --predictions runs/B1/predictions.jsonl --latency runs/B1/latency.jsonl \
    --env runs/B1/environment.json --hardware "NVIDIA L4"

# 3) 리더보드 + 그래프
python evaluation/make_leaderboard.py
```

예측 파일: jsonl, 한 줄에 `{"id": "<manifest id>", "text": "<모델 출력>"}`. Track A만 평가하려면 이것만 있으면 된다.
Track B는 `evaluation/stream_sim.py`의 `OfflineSystem` / `StreamingSystem` 인터페이스를 구현해 같은 시뮬레이터로 측정한다.

### 정규화 규칙 (`evaluation/normalizer.py`, v1.1)

1. NFC 정규화, `%` → `퍼센트`, 소수점 구어 표기 `쩜` → `점`
2. 아라비아 숫자 → 한자어 읽기 (`2026` → `이천이십육`, `1,000` → `천`, `3.5` → `삼점오`, 선행 0은 자리별 `031` → `공삼일`)
3. 영문자 대문자 통일
4. 문장부호·기호 제거, 연속 공백 1개로
5. CER은 공백까지 제거한 문자열로 계산

고유어 숫자(한, 두, 세)는 생성하지 않으므로, 숫자 표기가 섞인 원문은 발화 추출 단계에서 제외한다(정규화 안정성 필터).

### 지연 측정 프로토콜 (`evaluation/latency_config.json`)

- 청크 0.5 s를 실시간 속도로 공급. Silero VAD threshold 0.5, min silence 300 ms, speech pad 30 ms. 모든 시스템 동일.
- 평가 오디오는 발화 앞뒤에 1 s의 잡음이 붙어 있어 종료 검출을 잡음 속에서 해야 한다.
- 정답 `speech_start/speech_end`는 잡음 합성 전 클린 음성에 같은 VAD를 적용해 구한다.
- 최종 지연 = 최종 텍스트 시각 − 정답 종료 시각. 종료 검출 오차 = VAD 종료 시각 − 정답 종료 시각. VAD가 정답 종료 전에 끊으면 조기 절단.
- 오프라인: VAD 종료 시점까지 들어온 오디오 전체를 1회 추론(조기 절단 시 뒷부분은 잘린 채 평가). 스트리밍: 청크마다 `push()`, 부분결과 변화를 모두 기록.
- 부분결과 수정률: 이전 부분결과가 새 부분결과의 접두부가 아닌 갱신의 비율(공백 차이는 무시).
- 워밍업 3발화 제외, 발화당 3회 반복 중 최종 지연 중앙값 채택. RTFx = 오디오 길이 ÷ 처리 시간.

## 테스트

```bash
python -m pytest
```

`tests/test_stream_sim_integration.py`는 torch·silero-vad와 추출된 클린 wav가 있을 때만 실행된다.

## 실행 환경

Colab, NVIDIA L4 고정. GPU·CUDA·torch/transformers/vLLM/qwen-asr 버전은 각 결과 JSON의 `environment`에 기록.

## 데이터·라이선스

- 원본 음성: [zeroth-korean](https://huggingface.co/datasets/kresnik/zeroth_korean) test 250발화, [FLEURS](https://huggingface.co/datasets/google/fleurs) ko_kr test 174발화 (둘 다 CC BY 4.0)
- 잡음: MUSAN, FSD50K 중 CC0·CC BY 클립만 (파일별 라이선스는 `data/noise_list.csv`)
- AI Hub 자료는 재배포 금지라 사용하지 않음
- Qwen3-ASR 기술 보고서가 한국어를 FLEURS로 평가했으므로 Clean 점수는 기준점으로만 보고, 잡음 조건에서의 하락폭을 중심으로 해석한다.
- 베이스라인: [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) ([1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B))
