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
| B. 실시간성 | 동일 | 최종 지연 p50/p90 (s) ↓ · 종료 검출 오차 (s) ↓ · 조기 절단율 (%) ↓ · 부분결과 수정률 (%) ↓ (스트리밍) · 첫 부분결과 지연 (s) ↓ 및 발화 전 부분결과율 (%) ↓ (스트리밍) · RTFx ↑ |

순위: 잡음 4조건 평균 CER → 동률 시 평균 p90. 한 조건이라도 p90 > 1.0 s면 순위 밖 별도 표기.

## 레포 구조

```
baseline/
  run_qwen3_asr.py        Qwen3-ASR 1.7B/0.6B × offline/streaming (B1~B4)
  configs/B1..B4.json
data/
  eval/                   manifest_clean_<source>.jsonl (클린 중간), manifest.jsonl (최종), clean_subset_ids_<source>.txt
                          오디오(919 MB)는 HF Datasets: https://huggingface.co/datasets/ekgP115/campusvoice-rt-eval
  noise_list.csv          사용 잡음 클립 160개 + 개별 라이선스
  LICENSES/               FSD50K 출처 표기 (생성)
  scripts/
    select_subset.py      Zeroth·FLEURS test에서 발화 추출 + VAD 정답 구간
    filter_noise.py       FSD50K CC0·CC BY 필터, 4범주 선택, 클립만 다운로드
    p56.py                ITU-T P.56 활성 음성 레벨
    mix_noise.py          SNR 4조건 합성, 앞뒤 1 s 잡음 패딩, 시드 고정
    upload_hf.py          평가셋 Hugging Face Datasets 업로드
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
notebooks/reproduce_colab.ipynb   베이스라인 재현
notebooks/build_dataset_colab.ipynb  평가셋 생성(디스크 부족 시 Colab에서)
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

- 청크 0.5 s 단위 공급을 이벤트 시간으로 시뮬레이션한다. 청크 i는 스트림 시각 (i+1)×0.5 s에 도착하고, 가상 시계는 도착 시각까지 전진한 뒤 VAD·모델 호출의 실측 wall-clock만큼 더 전진한다. 처리가 공급을 따라가면 실시간 재생과 같고, 못 따라가면 대기 지연이 누적된다. sleep이 없어 실행 시간은 순수 연산 시간뿐이다.
- 엔드포인터는 Silero VAD threshold 0.5, **min silence 500 ms**, speech pad 30 ms. 32 ms 프레임 단위로 동작하므로 검출 시각은 청크 경계가 아니라 해당 프레임 도착 시각이다. 모든 시스템 동일.
- **ASR 공급은 VAD로 게이트된다.** 실제 음성 비서처럼 VAD가 발화 시작을 보고한 뒤에야 모델에 오디오를 넘기며, 넘기는 오디오는 시작 시점 200 ms 전(`preroll_ms`)부터 종료 검출 시점까지다. 스트리밍은 그때까지 버퍼된 오디오를 첫 push로 보내고 이후 0.5 s 청크를 이어 보낸다. VAD가 발화 시작을 못 잡으면 모델은 아무것도 받지 않는다(`missed`로 집계, 해당 발화 CER 100%).
- 게이트를 두는 이유: Qwen3-ASR 스트리밍은 앞 2청크 이후 이전 출력을 접두부로 고정하므로, 발화 전 무음·잡음 구간에서 만들어진 환각 텍스트가 고정되어 발화 전체가 엉뚱한 문장으로 대체되는 일이 생긴다(첫 GPU 점검에서 clean 조건 CER 18%, 같은 오디오의 20 dB 조건은 1.4%). 게이트 없이 흘려 넣는 것은 배포 구조와도 다르다.
- min silence를 500 ms로 둔 근거: 원본 424발화의 문장 내 최대 쉼은 p90 0.40 s, p95 0.50 s, 최대 1.5 s다. 300 ms면 clean에서도 15.6%가 조기 절단되고, 800 ms면 0.2%지만 종료 검출만으로 0.8 s를 써서 1.0 s 예산이 무의미해진다. 500 ms는 clean 조기 절단 5.0%를 감수하는 대신 실제 음성 비서 설정에 가깝고 제약이 변별력을 갖는다.
- 정답 `speech_start/end`를 구하는 VAD 설정(threshold 0.5, min silence 300 ms, pad 30 ms)은 `data/scripts/common.py`에 고정되어 엔드포인터 설정과 무관하다.
- 평가 오디오는 발화 앞뒤에 1 s의 잡음이 붙어 있어 종료 검출을 잡음 속에서 해야 한다.
- 정답 `speech_start/speech_end`는 잡음 합성 전 클린 음성에 같은 VAD를 적용해 구한다.
- 최종 지연 = 최종 텍스트 시각 − 정답 종료 시각. 종료 검출 오차 = VAD 종료 시각 − 정답 종료 시각. VAD가 정답 종료 전에 끊으면 조기 절단. 시작 검출 오차 = VAD 시작 시각 − 정답 시작 시각.
- 오프라인: VAD 종료 시점까지 들어온 오디오 전체를 1회 추론(조기 절단 시 뒷부분은 잘린 채 평가). 스트리밍: 청크마다 `push()`, 부분결과 변화를 모두 기록.
- 부분결과 수정률: 정규화(문장부호·공백 제거)한 이전 부분결과가 새 부분결과의 접두부가 아닌 갱신의 비율. 문장부호·공백만 바뀐 갱신은 세지 않는다.
- 발화 전 부분결과율: 첫 부분결과가 정답 발화 시작보다 먼저 나온 발화의 비율. 앞쪽 잡음·무음에서 텍스트를 만들어낸(환각) 정도를 보여준다.
- 평가 오디오의 앞뒤 1 s 패딩은 잡음 조건에서는 잡음, clean 조건에서는 -60 dBov 마이크 노이즈 플로어다(완전한 0은 환각을 유발해 제외).
- 워밍업 3발화(측정과 같은 시뮬레이터 경로로 실행, 결과 폐기) 제외. 전체 평가셋은 발화당 **1회 측정**한다. 가상 시계 방식에서는 wall-clock 흔들림이 모델 연산 시간에만 들어가고 2,120발화에 걸친 p50/p90은 그것에 둔감하므로, 반복 대신 B2(1.7B 스트리밍) 40발화를 3회 반복해 1회 측정과 3회 중앙값의 최종 지연 차이를 보고한다(`evaluation/jitter_check.py`, 결과는 리더보드 페이지에 명시). `latency_config.json`의 `repeats`를 올리면 3회 중앙값 방식으로 돌아간다. RTFx = 오디오 길이 ÷ 처리 시간.

## 베이스라인 (B1~B4, Qwen3-ASR, 학습 없음)

| ID | 모델 | 방식 | 비교 대상 |
|---|---|---|---|
| B1 | Qwen3-ASR-1.7B | 오프라인 | 정확도 상한 |
| B2 | Qwen3-ASR-1.7B | 스트리밍 | B1 대비 실시간 전환 효과 |
| B3 | Qwen3-ASR-0.6B | 오프라인 | B1 대비 모델 축소 효과 |
| B4 | Qwen3-ASR-0.6B | 스트리밍 | 지연 하한 |

- 백엔드는 네 시스템 모두 vLLM(`qwen-asr[vllm]`). qwen-asr의 스트리밍은 vLLM 전용이라 오프라인도 같은 백엔드로 맞췄다.
- 스트리밍 파라미터(`configs/B2.json`, `B4.json`): 모델 디코드 청크 1.0 s, `unfixed_chunk_num` 2, `unfixed_token_num` 5 (qwen-asr 기본값). 시뮬레이터의 공급 청크는 0.5 s로 고정이며 별개다.
- qwen-asr 스트리밍은 청크마다 지금까지의 오디오 전체를 다시 디코드하므로 발화가 길수록 한 스텝이 느려진다. 디코드가 공급 속도를 못 따라가면 지연이 누적되는데, 그것이 측정 대상이다.
- GPU 없이 파이프라인을 점검하려면 `--mock --limit 20 --repeats 1`.

## 결과 (NVIDIA L4, 2,120 평가 파일, 발화당 1회 측정)

| 시스템 | 잡음 평균 CER ↓ | CER clean / 20 / 10 / 5 / 0 dB | p90 최종 지연 clean → 0 dB | 종료 검출 오차 0 dB | 조기 절단 0 dB | 수정률 0 dB | 발화 전 부분결과 0 dB | RTFx | 제약 |
|---|---|---|---|---|---|---|---|---|---|
| B1 Qwen3-ASR-1.7B 오프라인 | **15.32** | 10.3 / 9.9 / 12.5 / 16.0 / 22.9 | 1.33 → 1.59 | 0.560 | 15.1% | - | - | 22.4 | 초과 (p90 max 1.59 s) |
| B2 Qwen3-ASR-1.7B 스트리밍 | **16.42** | 10.8 / 10.8 / 13.3 / 17.2 / 24.3 | 0.81 → 1.14 | 0.560 | 15.1% | 38.4 | 6.8 | 7.7 | 초과 (p90 max 1.14 s) |
| B3 Qwen3-ASR-0.6B 오프라인 | **17.95** | 11.5 / 11.4 / 14.5 / 19.1 / 26.9 | 0.87 → 1.24 | 0.560 | 15.1% | - | - | 52.8 | 초과 (p90 max 1.24 s) |
| B4 Qwen3-ASR-0.6B 스트리밍 | **18.92** | 12.0 / 12.0 / 15.0 / 20.4 / 28.4 | 0.67 → 1.08 | 0.560 | 15.1% | 34.1 | 6.8 | 14.8 | 초과 (p90 max 1.08 s) |

- 네 시스템 모두 잡음 4조건 중 최소 한 조건에서 p90 최종 지연이 1.0 s를 넘어 **메인 순위는 비어 있다**. 가장 가까운 것은 B4(0 dB에서만 1.08 s 초과)이고, B2는 5 dB부터, B3는 10 dB부터, B1은 clean에서도 넘는다.
- 잡음 평균 CER 순서는 B1 < B2 < B3 < B4로 모델 크기·오프라인이 유리하지만, 지연은 정반대다. 스트리밍은 종료 검출 시점에 대부분을 이미 디코드해 둔 상태라 종료 후 처리가 0.07~0.14 s인 반면, 오프라인은 종료 후 전체 추론(B1 p90 0.80 s)을 치른다.
- 흔들림 검증(B2 40발화 × 5조건, 3회 반복 중앙값 대비): 최종 지연 차이 중앙값 29 ms, p90 0.866 s vs 0.831 s. 1회 측정으로 충분하다.

### CER의 어디까지가 잡음 때문인가

| 시스템 | CER 전체 clean → 0 dB | CER 절단 제외 clean → 0 dB | 종료 검출 지연 p90 clean → 0 dB | 종료 후 처리 p90 | p90 최종 지연 clean → 0 dB |
|---|---|---|---|---|---|
| B1 | 10.3 → 22.9 | 2.5 → 12.2 | 0.56 → 0.99 s | 0.80 s | 1.33 → 1.59 s |
| B2 | 10.8 → 24.3 | 3.0 → 14.0 | 0.56 → 0.99 s | 0.29 s | 0.81 → 1.14 s |
| B3 | 11.5 → 26.9 | 3.8 → 16.8 | 0.56 → 0.99 s | 0.34 s | 0.87 → 1.24 s |
| B4 | 12.0 → 28.4 | 4.3 → 18.5 | 0.56 → 0.99 s | 0.15 s | 0.67 → 1.08 s |

- 엔드포인터는 네 시스템이 공유하므로 종료 검출 오차·조기 절단율은 시스템과 무관하게 같다(조기 절단 발화 수: clean 43, 20 dB 39, 10 dB 45, 5 dB 53, 0 dB 64). 절단된 발화는 뒷부분이 통째로 빠져 CER이 66~75%에 이르고, 이것이 전체 CER의 큰 몫을 차지한다.
- 절단되지 않은 발화만 보면 clean CER은 2.5~4.3%로 Qwen3-ASR 기술 보고서의 FLEURS 수치와 비슷하고, 0 dB에서 12~19%로 오른다. 즉 **잡음은 인식 오류를 4~5배 늘리는 동시에 조기 절단 발화를 43개에서 64개로 늘려** CER을 두 경로로 올린다.
- 조기 절단은 낭독 문장의 긴 쉼(min silence 500 ms 초과)에서 생기며, 짧은 질의 중심의 실제 음성 비서 입력에서는 덜 나타날 것이다. 리더보드 CER은 사용자가 겪는 그대로(절단 포함)를 쓰고, 절단 제외 CER은 `results/analysis.json`에 함께 제공한다.
- 그래프: `results/latency_cer_paths.png` (시스템별 Clean → 0 dB 이동 경로).

## 테스트

```bash
python -m pytest
```

`tests/test_stream_sim_integration.py`는 torch·silero-vad와 추출된 클린 wav가 있을 때만 실행된다.

## 실행 환경

Colab, NVIDIA L4 고정. GPU·CUDA·torch/transformers/vLLM/qwen-asr 버전은 각 결과 JSON의 `environment`에 기록.

## 데이터·라이선스

- 원본 음성: [zeroth-korean](https://huggingface.co/datasets/kresnik/zeroth_korean) test 250발화, [FLEURS](https://huggingface.co/datasets/google/fleurs) ko_kr test 174발화 (둘 다 CC BY 4.0)
- 잡음: FSD50K 중 CC0·CC BY 클립만 (클립별 라이선스는 `data/noise_list.csv`, 출처 표기는 `data/LICENSES/FSD50K.md`)
- AI Hub 자료는 재배포 금지라 사용하지 않음
- Qwen3-ASR 기술 보고서가 한국어를 FLEURS로 평가했으므로 Clean 점수는 기준점으로만 보고, 잡음 조건에서의 하락폭을 중심으로 해석한다.
- 베이스라인: [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) ([1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B))
