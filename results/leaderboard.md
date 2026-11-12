## Main ranking (p90 final latency ≤ 1.0 s in every noisy condition)

| # | System | Model | Mode | Endpointer | Noisy CER (%) ↓ | Clean CER | CER 20/10/5/0 dB | Noisy p90 avg (s) ↓ | p90 max (s) | Endpoint err 0 dB (s) | Early cut 0 dB (%) | Revision 0 dB (%) | RTFx ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S1 | Qwen/Qwen3-ASR-0.6B | streaming | custom:participants.adaptive_silero.AdaptiveSileroEndpointer | 17.20 | 9.44 | 10.1/12.7/17.8/28.3 | 0.761 | 0.830 | 0.504 | 17.0 | 35.2 | 9.7 |
| 2 | S2 | Qwen/Qwen3-ASR-1.7B | streaming | custom:participants.adaptive_silero.AdaptiveSileroEndpointer | 18.46 | 8.44 | 9.1/11.7/14.8/38.2 | 0.881 | 0.955 | 0.492 | 17.7 | 39.5 | 5.0 |

## Outside the latency constraint / no Track B measurement

| # | System | Model | Mode | Endpointer | Noisy CER (%) ↓ | Clean CER | CER 20/10/5/0 dB | Noisy p90 avg (s) ↓ | p90 max (s) | Endpoint err 0 dB (s) | Early cut 0 dB (%) | Revision 0 dB (%) | RTFx ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | B1 | Qwen/Qwen3-ASR-1.7B | offline | default | 15.32 | 10.31 | 9.9/12.5/16.0/22.9 | 1.490 | 1.593 | 0.560 | 15.1 | - | 22.4 |
| - | B2 | Qwen/Qwen3-ASR-1.7B | streaming | default | 16.42 | 10.77 | 10.8/13.3/17.2/24.3 | 0.979 | 1.143 | 0.560 | 15.1 | 38.4 | 7.7 |
| - | B3 | Qwen/Qwen3-ASR-0.6B | offline | default | 17.95 | 11.53 | 11.4/14.5/19.1/26.9 | 1.070 | 1.244 | 0.560 | 15.1 | - | 52.8 |
| - | B4 | Qwen/Qwen3-ASR-0.6B | streaming | default | 18.92 | 12.01 | 12.0/15.0/20.4/28.4 | 0.887 | 1.084 | 0.560 | 15.1 | 34.1 | 14.8 |
