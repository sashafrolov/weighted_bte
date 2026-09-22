# MCL arithmetic kernel results

Single CPU; median of each independent round's sample median. Times are milliseconds.
BN254 below is **MCL BN_SNARK1**, not MCL's separately named BN254.
These rows are operation-shape probes and are not end-to-end protocol results.

## Actual AVX-512 dispatch

| B | G1 operation | SIMD off ms | AVX-512 ms | Paired speedup range | MSM / mulEach calls per repetition |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 | partial_decrypt_one_msm | 0.0846 | 0.0845 | 1.00–1.00× | 0 / 0 |
| 4 | committee_positive_msms | 30.3447 | 15.5621 | 1.95–1.95× | 3 / 0 |
| 4 | group_fft | 0.1445 | 0.1442 | 1.00–1.00× | 0 / 0 |
| 16 | partial_decrypt_one_msm | 0.2666 | 0.2666 | 1.00–1.00× | 0 / 0 |
| 16 | committee_positive_msms | 151.6379 | 77.7905 | 1.95–1.95× | 15 / 0 |
| 16 | group_fft | 1.6952 | 1.6928 | 1.00–1.00× | 0 / 0 |
| 64 | partial_decrypt_one_msm | 1.0565 | 1.0567 | 1.00–1.00× | 0 / 0 |
| 64 | committee_positive_msms | 636.6782 | 326.5639 | 1.95–1.95× | 63 / 0 |
| 64 | group_fft | 11.8103 | 4.7066 | 2.51–2.51× | 0 / 6 |

Rows with zero dispatch are controls, not SIMD speedups.

## Same-backend curve comparisons (SIMD off)

| B | Operation | Group | BLS12-381 ms | BN_SNARK1 ms | BLS / BN paired ratio |
| --- | --- | --- | ---: | ---: | ---: |
| 4 | partial_decrypt_one_msm | G1 | 0.0846 | 0.0515 | 1.64–1.64× |
| 4 | committee_opening_msms | G1 | 55.4432 | 33.5416 | 1.65–1.65× |
| 4 | committee_opening_msms | G2 | 135.1305 | 72.7039 | 1.86–1.86× |
| 4 | committee_positive_msms | G1 | 30.3447 | 17.6917 | 1.71–1.72× |
| 4 | committee_positive_msms | G2 | 55.2203 | 29.6709 | 1.86–1.86× |
| 4 | opening_multi_pairings | G1xG2 | 16.2630 | 12.1282 | 1.34–1.34× |
| 16 | partial_decrypt_one_msm | G1 | 0.2666 | 0.1605 | 1.66–1.66× |
| 16 | committee_opening_msms | G1 | 222.1704 | 134.1751 | 1.65–1.66× |
| 16 | committee_opening_msms | G2 | 542.6709 | 290.2233 | 1.87–1.87× |
| 16 | committee_positive_msms | G1 | 151.6379 | 88.4709 | 1.71–1.72× |
| 16 | committee_positive_msms | G2 | 277.0478 | 148.3123 | 1.87–1.87× |
| 16 | opening_multi_pairings | G1xG2 | 65.0362 | 48.6834 | 1.34–1.34× |
| 64 | partial_decrypt_one_msm | G1 | 1.0565 | 0.6257 | 1.69–1.69× |
| 64 | committee_opening_msms | G1 | 887.0113 | 537.2109 | 1.65–1.65× |
| 64 | committee_opening_msms | G2 | 2158.7549 | 1159.6955 | 1.86–1.86× |
| 64 | committee_positive_msms | G1 | 636.6782 | 371.8140 | 1.71–1.71× |
| 64 | committee_positive_msms | G2 | 1165.0446 | 628.3966 | 1.85–1.86× |
| 64 | opening_multi_pairings | G1xG2 | 260.8678 | 194.6178 | 1.33–1.35× |
