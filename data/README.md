# Benchmark Dataset

| 구성 | 소스 | 조건 | 라이선스 | 용도 |
|---|---|---|---|---|
| 클린 기준셋 | [zeroth-korean](https://huggingface.co/datasets/kresnik/zeroth_korean) test 분할 200발화, 화자 균등 추출 (`scripts/build_clean_subset.py`) | clean + 4 SNR | CC BY 4.0 | 일반 발화 기준 |
| 캠퍼스 도메인셋 (TTS) | `sentences/` 약 200문장 → TTS 합성 | clean + 4 SNR | 자체 제작 | 핵심어 인식 |
| 캠퍼스 도메인셋 (실녹음) | 동일 문장 중 50~100문장 본인 녹음 | clean + 4 SNR | 자체 제작 | 합성 vs 실발화 비교 |

잡음: MUSAN (noise/babble), 잔향: OpenSLR SLR28 RIR. SNR 20/10/5/0 dB, 시드 고정 (`scripts/augment_noise_rir.py`).

## 파일

- `eval/manifest.jsonl` — 최종 평가셋 manifest (아래 형식, clean + 4 SNR 조건). 오디오 원본은 Hugging Face Datasets에 업로드 (링크 TBD).
- `eval/manifest_clean_zeroth.jsonl`, `eval/manifest_clean_domain.jsonl` — 잡음 합성 전 클린 중간 manifest. `scripts/augment_noise_rir.py`가 이들을 합쳐 최종 manifest를 만든다.
- `eval/clean_subset_ids.txt` — zeroth-korean test 분할에서 고른 발화 id 목록. 시드와 무관하게 이 목록이 기준이며, 재실행 시 그대로 재사용된다.
- `terms.txt` — Term Recall 용어 사전.
- `sentences/` — 도메인 문장 원문 + 출처 URL.
- `scripts/` — 클린 서브셋 추출, TTS 합성, 잡음·잔향 합성.

## manifest 형식

```json
{"id": "dom_0001", "audio": "audio/dom_0001.wav", "duration": 3.42, "text": "휴학 신청은 언제까지 해야 하나요", "terms": ["휴학"], "domain": "행정절차", "source": "tts", "condition": "clean", "speech_start": 0.21, "speech_end": 3.10}
```

- `condition`: `clean` | `snr20` | `snr10` | `snr5` | `snr0`
- `source`: `tts` | `recorded` | 코퍼스 이름
- `terms`: `evaluation.metrics.find_terms(text, terms)`로 생성
- `speech_start` / `speech_end`: 원본(클린) 오디오 기준 발화 구간 (초). 지연 측정 기준점.
