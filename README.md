# Implementations for Weighted Batch Threshold Encryption paper

Directory structure:
- btx: Our reimplementation of the BTX encryption scheme.
- pfe: Our reimplementation of the Boneh et al. partial fraction-based batch threshold encryption scheme.
- scripts: various scripts used in this paper, including our benchmark parameter sweep scripts, and scripts for pulling and working with Solana's distributions of shares.
- simple-bte: Policharla's implementation of the BTX scheme.
- weighted_btx: Implementation of our new scheme (weighted version of BTX)
- weighted_indexed_bte: Our reimplementation of the prior work with weighted indexed BTE.
- weighted_pfe: Implementation of our new scheme (weighted version of PFE)

Experimental variants and benchmarks:
- [weighted_btx_swapped](weighted_btx_swapped/README.md): G1/G2-swapped weighted BTX over BLS12-381.
- [weighted_btx_mcl](weighted_btx_mcl/README.md): Complete in-memory protocol over BLS12-381 and BN254, with optional BLS G1 AVX-512 acceleration.
- [experiments](experiments/README.md): Correctness-checked benchmarks for 1x16, 4x4, and 8x2 layouts, including reusable committee preparation and full online timings.
