# Primitive Roofline Workspace

This folder is a clean workspace for primitive-level roofline and performance analysis.

## Layout

- `profiles/`
  - Machine profiles (CPU + DRAM parameters).
- `primitives/`
  - One folder per primitive to document compute/memory behavior.
- `experiments/`
  - Input CSV templates and run manifests.
- `results/raw/`
  - Raw benchmark outputs.
- `results/processed/`
  - Derived metrics and merged tables.
- `results/plots/`
  - Roofline plots and figures.
- `scripts/`
  - Utility scripts for roofline calculation and primitive analysis.
- `src/`
  - Primitive benchmark source (`primitive_bench.cpp`).
- `bin/`
  - Local benchmark executable (`primitive_bench`).

## Five Primitive Suite

The suite currently includes:

1. `gemv`
2. `cos_db_db`
3. `cos_q_db`
4. `ip_q_db`
5. `softmax`

Model definitions (FLOPs/Bytes/AI):

- `primitives/FIVE_PRIMITIVES_MODEL_FP32.md`

Softmax in this workspace uses a fused-kernel traffic assumption for AI
comparison (no repeated intermediate read/write in the model).

## Quick Start

1. Build local primitive benchmark:
   - `make bench`
2. Compute machine roofline constants:
   - `make roofline-fp32`
3. Run the five-primitive suite:
   - `make run-suite`
4. Summarize against roofline:
   - `make summarize-suite`
5. One-shot run:
   - `make suite-fp32`
6. Plot roofline line + primitive scatter:
   - `make plot-roofline`

Hardware-simulator style path (perf-based PMU approximation):

1. Run model suite:
   - `make run-suite`
2. Run hardware PMU suite:
   - `make run-suite-hw`
3. Merge roofline + model + hardware simulation:
   - `make summarize-suite-hw`
4. One-shot:
   - `make suite-fp32-hw`
5. One-shot with plot:
   - `make suite-fp32-hw-plot`

Pin thread count if needed (example: 36 threads):

- `make run-suite THREADS=36`
- `make run-suite-hw THREADS=36`
- `make summarize-suite-hw`
- `make plot-roofline THREADS=36`

## End-to-End Script

You can run the full experiment pipeline (build + roofline + suite + hw-sim + summary + plot) with one command:

- `bash scripts/run_end_to_end_fp32.sh --threads 36 --seed 42`

Optional flags:

- `--tag <name>`: custom run tag for output filenames
- `--show-hw-in-plot`: also overlay `hw_fp_gflops` as `x` markers

Equivalent Make target:

- `make e2e-fp32 THREADS=36 SEED=42 RUN_TAG=my_run`

Default suite config:

- `configs/suite_fp32.csv`

Suite config schema (`run_suite.py` / `run_suite_hw_sim.py`):

- Preferred columns:
  - `workload`
  - `params_json` (JSON object of CLI params, for example `{"dim":128,"scale":262144}`)
  - `target_seconds`
- Optional per-row overrides:
  - `seed`
  - `threads`
- Backward compatibility:
  - legacy columns `primitive,dim,scale,db_vectors,target_seconds` are still supported.

Default outputs:

- raw benchmark rows: `results/raw/suite_fp32_raw.csv`
- roofline summary: `results/processed/suite_fp32_summary.csv`
- hardware simulation rows: `results/raw/suite_fp32_hw_sim.csv`
- merged hardware summary: `results/processed/suite_fp32_summary_hw.csv`
- roofline plot: `results/plots/roofline_fp32_primitives.svg`

## Measurement CSV Format

`experiments/measurements_template.csv` requires:

- `primitive`: primitive or operation name
- `ai`: arithmetic intensity (FLOP/Byte)
- `achieved_gflops`: measured throughput (GFLOP/s)
- `notes`: optional remarks

## Notes

- `scripts/run_suite.py` auto-runs to target stable runtime per primitive.
- `scripts/run_suite.py` and `scripts/run_suite_hw_sim.py` collect structured benchmark outputs via `primitive_bench --json`.
- `scripts/run_suite_hw_sim.py` runs PMU events via `perf stat` for hardware-level approximation.
- `scripts/plot_roofline.py` draws roofline line + primitive scatter points from summary CSV.
- `scripts/run_end_to_end_fp32.sh` is the one-click end-to-end experiment entry.
- `src/primitive_bench.cpp` supports `--fixed-repetitions` for deterministic replay of workload size.
- PMU FLOP counts reflect executed floating-point arithmetic (including transcendental implementation internals), so they can exceed algorithm-model FLOP counts.
- PMU events are process-wide samples from `perf stat`; hardware metrics should be interpreted as approximation, not exact kernel-only counters.
- Keep raw data in `results/raw/` and derived data in `results/processed/`.
- `scripts/scaffold_primitive.sh` can create a new primitive folder with templates.
