# Benchmark Dataset

| 구성 | 소스 | 규모 | 라이선스 |
|---|---|---|---|
| 원본 음성 | [zeroth-korean](https://huggingface.co/datasets/kresnik/zeroth_korean) test, 10화자 균등 (`scripts/select_subset.py`) | 250발화 | CC BY 4.0 |
| 원본 음성 | [FLEURS](https://huggingface.co/datasets/google/fleurs) ko_kr test, 문장당 1녹음, 숫자·영문 포함 문장 제외, 20 s 이하 (`scripts/select_subset.py --source fleurs`) | 174발화 (남 125 / 여 49) | CC BY 4.0 |
| 잡음 | MUSAN, FSD50K CC0·CC BY 클립 — 군중·대화, 바람, 차량, 야외 행사 (`scripts/filter_noise.py`) | TBD | 파일별 (`noise_list.csv`) |
| 평가셋 | 원본 424발화 × {clean, snr20, snr10, snr5, snr0} (`scripts/mix_noise.py`) | 2,120파일 | — |

합성: 16 kHz mono, ITU-T P.56 활성 음성 레벨 기준 SNR, 발화 앞뒤 1 s 잡음 패딩, 잡음 종류·파일·오프셋 시드 고정 무작위 배정, 클리핑 검사. 야외라 잔향은 넣지 않는다.

## 파일

- `eval/manifest_clean_zeroth.jsonl`, `eval/manifest_clean_fleurs.jsonl` — 잡음 합성 전 클린 중간 manifest
- `eval/manifest.jsonl` — 최종 평가셋 manifest (아래 형식). 오디오는 Hugging Face Datasets에 업로드 (링크 TBD)
- `eval/clean_subset_ids_zeroth.txt`, `eval/clean_subset_ids_fleurs.txt` — 선택된 원본 발화 id. 재실행 시 그대로 재사용
- FLEURS test는 382녹음/270문장인데 숫자(86)·영문(61) 포함 문장을 빼고 문장당 1개만 남기면 174개가 상한이라 "약 250"에서 줄였다. 숫자는 고유어/한자어 읽기를 전사만으로 알 수 없고, 영문은 괄호 병기라 실제 발화와 다르기 때문이다.
- `noise_list.csv`, `LICENSES/` — 잡음 파일 목록과 출처 표기

## manifest 형식

```json
{"id": "zeroth_0123_snr5", "audio": "audio/snr5/zeroth_0123.wav", "source": "zeroth", "source_id": "104_003_0019", "speaker": "zeroth_104", "text": "...", "condition": "snr5", "noise": {"type": "crowd", "file": "fsd50k/123456.wav", "offset": 12.4, "snr_db": 5}, "speech_start": 1.70, "speech_end": 9.60, "duration": 11.93, "seed": 2026}
```

- `condition`: `clean` | `snr20` | `snr10` | `snr5` | `snr0`
- `speech_start` / `speech_end`: 클린 원본에 Silero VAD를 적용해 구한 발화 구간(초)에 패딩 1 s를 더한 값. 지연·종료 검출 지표의 기준점
- `noise`: clean 조건에서는 null
