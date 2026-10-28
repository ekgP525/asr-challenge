# Benchmark Dataset

| 구성 | 소스 | 규모 | 라이선스 |
|---|---|---|---|
| 원본 음성 | [zeroth-korean](https://huggingface.co/datasets/kresnik/zeroth_korean) test, 10화자 균등 (`scripts/select_subset.py`) | 250발화 | CC BY 4.0 |
| 원본 음성 | [FLEURS](https://huggingface.co/datasets/google/fleurs) ko_kr test, 문장당 1녹음, 숫자·영문 포함 문장 제외, 20 s 이하 (`scripts/select_subset.py --source fleurs`) | 174발화 (남 125 / 여 49) | CC BY 4.0 |
| 잡음 | [FSD50K](https://zenodo.org/record/4060432) CC0·CC BY 클립만, 4범주 × 40클립 — crowd(군중·대화), wind(바람), traffic(차량), event(환호·박수) (`scripts/filter_noise.py`) | 160클립, 약 35분 | 클립별 (`noise_list.csv`, `LICENSES/FSD50K.md`) |
| 평가셋 | 원본 424발화 × {clean, snr20, snr10, snr5, snr0} (`scripts/mix_noise.py`) | 2,120파일 | — |

합성 (`scripts/mix_noise.py`): 16 kHz mono. 먼저 모든 원본 발화를 ITU-T P.56 활성 음성 레벨(`scripts/p56.py`)이 -26 dBov가 되도록 정규화한다(원본은 -65~-12 dBov로 편차가 커서 양자화·VAD 반응의 교란 요인이 됨; 피크가 0.99를 넘으면 그 한도까지만 올리고 `speech_gain`에 기록). 잡음 레벨은 실제 사용한 잡음 구간의 RMS이고, 정규화된 음성 레벨과의 차가 SNR이 되도록 잡음 이득을 정한다. 발화 앞뒤 1 s는 clean 조건에서는 무음, 잡음 조건에서는 잡음이 이어진다. 발화마다 잡음 범주·클립·시작 오프셋을 시드 고정 무작위로 정하고, 같은 발화의 4개 SNR은 같은 잡음을 레벨만 바꿔 쓴다. 짧은 클립은 50 ms 크로스페이드로 이어 붙이고, 클리핑이 나면 혼합 전체를 줄이고 `gain`에 기록한다. 야외라 잔향은 넣지 않는다.

잡음 클립은 Hugging Face 미러 `Fhrozen/FSD50k`에서 필요한 클립만 받는다(전체 32 GB 불필요). MUSAN은 범주 라벨이 없어 쓰지 않는다. AI Hub 자료는 재배포 금지라 쓰지 않는다.

## 파일

- `eval/manifest_clean_zeroth.jsonl`, `eval/manifest_clean_fleurs.jsonl` — 잡음 합성 전 클린 중간 manifest
- `eval/manifest.jsonl` — 최종 평가셋 manifest (아래 형식). 오디오는 Hugging Face Datasets에 업로드 (링크 TBD)
- `eval/clean_subset_ids_zeroth.txt`, `eval/clean_subset_ids_fleurs.txt` — 선택된 원본 발화 id. 재실행 시 그대로 재사용
- FLEURS test는 382녹음/270문장인데 숫자(86)·영문(61) 포함 문장을 빼고 문장당 1개만 남기면 174개가 상한이라 "약 250"에서 줄였다. 숫자는 고유어/한자어 읽기를 전사만으로 알 수 없고, 영문은 괄호 병기라 실제 발화와 다르기 때문이다.
- `noise_list.csv`, `LICENSES/FSD50K.md` — 잡음 클립 목록(Freesound id, 제목, 업로더, 라이선스, 길이)과 출처 표기. 오디오 자체는 `data/noise/`에 생성되며 git에는 넣지 않는다
- `eval/audio/src/` — 패딩 전 원본 발화(선택 스크립트 출력), `eval/audio/{clean,snr20,snr10,snr5,snr0}/` — 최종 평가 파일

## manifest 형식

```json
{"id": "zeroth_0123_snr5", "audio": "audio/snr5/zeroth_0123.wav", "condition": "snr5", "noise": {"type": "crowd", "file": "fsd50k/123456.wav", "freesound_id": "123456", "offset": 12.4, "snr_db": 5, "noise_gain": 0.0494}, "gain": 1.0, "source": "zeroth", "source_id": "104_003_0019", "speaker": "zeroth_104", "text": "...", "duration": 11.93, "speech_start": 1.70, "speech_end": 9.60, "seed": 2026, "speech_level_dbov": -26.0, "source_level_dbov": -31.2, "speech_gain": 1.82, "activity": 0.71}
```

- `condition`: `clean` | `snr20` | `snr10` | `snr5` | `snr0`
- `speech_start` / `speech_end`: 클린 원본에 Silero VAD를 적용해 구한 발화 구간(초)에 패딩 1 s를 더한 값. 지연·종료 검출 지표의 기준점
- `noise`: clean 조건에서는 null. `noise_gain`은 잡음 클립에 곱한 이득, `gain`은 클리핑 방지를 위해 혼합 전체에 곱한 이득(1.0이면 미적용)
- `speech_level_dbov`: 정규화 후 P.56 활성 레벨, `source_level_dbov`: 원본 레벨, `speech_gain`: 정규화 이득, `activity`: P.56 활성도
