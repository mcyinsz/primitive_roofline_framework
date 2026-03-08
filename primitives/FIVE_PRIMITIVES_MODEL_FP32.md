# Five Primitive Models (FP32)

This document defines the FLOP and byte models used by `src/primitive_bench.cpp`.

## 1) GeMV (regular access)

- Operation: `y = A * x`
- Tunable parameter: reduction dimension `K` (`--dim`)
- Scale dimension: non-reduction matrix dimension `M` (`--scale`)

Per repetition model:

- FLOPs: `2 * M * K`
- Bytes: `4 * (M*K + K + M)`
- AI:
  - `AI = (2*M*K) / (4*(M*K + K + M))`

## 2) Cosine(v_i, v_j), both random from DB (random access)

- Tunable parameter: vector dimension `D` (`--dim`)
- Scale dimension: number of random pairs `P` (`--scale`)

Per pair model:

- FLOPs: `6*D + 3` (dot + two norms + scalar ops)
- Bytes: `8*D + 4` (two vector reads + one score write)
- AI:
  - `AI = (6*D + 3) / (8*D + 4)`

Per repetition model multiplies by `P`.

## 3) Cosine(q, v_i), one fixed query + random DB vectors (random access)

- Tunable parameter: vector dimension `D` (`--dim`)
- Scale dimension: number of random vectors `N` (`--scale`)

Per compared vector model:

- FLOPs: `4*D + 1` (dot + one norm + scalar division)
- Bytes: `4*D + 4` (one random vector read + one score write)
- AI:
  - `AI = (4*D + 1) / (4*D + 4)`

Per repetition model multiplies by `N` (plus one-time query load).

## 4) InnerProduct(q, v_i), one fixed query + random DB vectors (random access)

- Tunable parameter: vector dimension `D` (`--dim`)
- Scale dimension: number of random vectors `N` (`--scale`)

Per compared vector model:

- FLOPs: `2*D`
- Bytes: `4*D + 4` (one random vector read + one score write)
- AI:
  - `AI = (2*D) / (4*D + 4)`

Per repetition model multiplies by `N` (plus one-time query load).

## 5) Softmax(x)

- Tunable parameter: none
- Scale dimension: vector length `D` (`--scale`)

Implemented benchmark kernel:

1. pass 1: parallel max reduction (`omp for simd`)
2. pass 2: parallel `exp + sum` reduction, writing temporary exp values
3. pass 3: parallel normalization (`omp for simd`)

Per repetition model:

- FLOPs: `4*D` (model-level approximation)
- Bytes: `8*D` under fused-kernel assumption (input read + output write)
- AI:
  - `AI = (4*D) / (8*D) = 0.5`

## Note on model usage

- These are analytic model FLOPs/Bytes for roofline comparison.
- Real microarchitectural execution (e.g., `exp`) may differ in instruction mix and latency.
- Keep the same model for apples-to-apples comparison across runs.
