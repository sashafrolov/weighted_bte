# Full online protocol results

Milliseconds per 16 fresh ciphertexts, median [p10–p90]. 30 samples per case across 2 rounds (r1, r2; 15 samples per round). Descriptive quantiles are not confidence intervals.

The matrix runner reverses implementation order between odd and even rounds.

## end_to_end

| Backend / curve / SIMD / groups | Cold 1×16 | Cold 4×4 | Cached 1×16 | Cached 4×4 |
| --- | ---: | ---: | ---: | ---: |
| blst / bls12_381 / native / normal | 154.881 [153.083–159.363] | 66.320 [65.823–68.291] | 28.632 [28.580–28.835] | 20.409 [20.268–20.586] |
| blst / bls12_381 / native / swapped | 84.262 [81.869–85.376] | 42.187 [41.642–43.105] | 32.911 [32.838–33.080] | 22.938 [22.717–23.460] |
| mcl / bls12_381 / avx512 / normal | 128.028 [127.693–128.242] | 58.317 [58.068–58.742] | 22.862 [22.786–22.945] | 19.620 [19.509–19.721] |
| mcl / bls12_381 / avx512 / swapped | 64.295 [64.206–64.714] | 35.257 [35.185–35.351] | 24.849 [24.776–24.999] | 20.731 [20.650–20.823] |
| mcl / bls12_381 / off / normal | 128.963 [128.185–129.552] | 58.752 [58.402–59.217] | 22.821 [22.766–22.888] | 19.749 [19.627–19.816] |
| mcl / bls12_381 / off / swapped | 75.229 [75.101–75.647] | 40.343 [40.035–40.512] | 24.706 [24.643–24.871] | 20.751 [20.636–20.826] |
| mcl / bn254 / off / normal | 75.012 [74.716–75.217] | 36.136 [35.916–36.441] | 16.471 [16.316–16.715] | 14.141 [14.022–14.256] |
| mcl / bn254 / off / swapped | 50.527 [50.350–51.655] | 28.628 [28.507–28.890] | 19.248 [19.196–19.294] | 16.397 [16.312–16.523] |

## combiner

| Backend / curve / SIMD / groups | Cold 1×16 | Cold 4×4 | Cached 1×16 | Cached 4×4 |
| --- | ---: | ---: | ---: | ---: |
| blst / bls12_381 / native / normal | 151.661 [149.872–156.133] | 63.005 [62.439–64.883] | 25.417 [25.379–25.565] | 17.125 [16.932–17.192] |
| blst / bls12_381 / native / swapped | 78.281 [75.912–79.363] | 35.284 [35.022–36.120] | 26.993 [26.928–27.130] | 16.095 [16.046–16.147] |
| mcl / bls12_381 / avx512 / normal | 124.947 [124.649–125.146] | 54.958 [54.706–55.386] | 19.786 [19.708–19.868] | 16.226 [16.150–16.317] |
| mcl / bls12_381 / avx512 / swapped | 59.274 [59.199–59.660] | 29.924 [29.850–30.006] | 19.819 [19.755–19.954] | 15.360 [15.290–15.472] |
| mcl / bls12_381 / off / normal | 125.879 [125.109–126.493] | 55.371 [55.030–55.811] | 19.724 [19.692–19.795] | 16.335 [16.218–16.398] |
| mcl / bls12_381 / off / swapped | 70.190 [70.073–70.577] | 34.999 [34.724–35.162] | 19.696 [19.649–19.818] | 15.378 [15.305–15.436] |
| mcl / bn254 / off / normal | 72.850 [72.547–73.052] | 33.785 [33.579–34.115] | 14.317 [14.184–14.568] | 11.795 [11.702–11.920] |
| mcl / bn254 / off / swapped | 47.225 [47.047–48.013] | 25.192 [25.089–25.398] | 15.983 [15.923–16.015] | 12.923 [12.846–13.026] |

## SIMD dispatch

Counts are observed calls during the full timed online pipeline, excluding setup and warmups.

| Curve / SIMD / groups / layout / cache | MSM calls per iteration | Batched scalar calls per iteration |
| --- | ---: | ---: |
| bls12_381 / avx512 / normal / 1x16 / cached | 0–0 | 0–0 |
| bls12_381 / avx512 / normal / 1x16 / cold | 0–0 | 0–0 |
| bls12_381 / avx512 / normal / 4x4 / cached | 0–0 | 0–0 |
| bls12_381 / avx512 / normal / 4x4 / cold | 0–0 | 0–0 |
| bls12_381 / avx512 / swapped / 1x16 / cached | 0–0 | 0–0 |
| bls12_381 / avx512 / swapped / 1x16 / cold | 15–15 | 1–1 |
| bls12_381 / avx512 / swapped / 4x4 / cached | 0–0 | 0–0 |
| bls12_381 / avx512 / swapped / 4x4 / cold | 3–3 | 0–0 |

All raw per-phase samples, correctness logs, and build metadata accompany this summary. Setup and networking are excluded; all selected validators are simulated on one host with 12 execution threads.
