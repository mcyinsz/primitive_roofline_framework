PYTHON ?= python3
CXX ?= g++
CXXFLAGS ?= -O3 -march=native -ffast-math -fno-math-errno -fopenmp -fopenmp-simd -std=c++17
THREADS ?= 0
SEED ?= 42
RUN_TAG ?=
SHOW_HW_IN_PLOT ?= 0

PROFILE_FP32_OUT := results/processed/roofline_fp32.json
PROFILE_FP64_OUT := results/processed/roofline_fp64.json
MEASUREMENTS := experiments/measurements_template.csv
ANALYSIS_FP32_OUT := results/processed/analysis_fp32.csv
BENCH_SRC := src/primitive_bench.cpp
BENCH_BIN := bin/primitive_bench
SUITE_CFG := configs/suite_fp32.csv
SUITE_RAW_OUT := results/raw/suite_fp32_raw.csv
SUITE_SUMMARY_OUT := results/processed/suite_fp32_summary.csv
SUITE_HW_SIM_OUT := results/raw/suite_fp32_hw_sim.csv
SUITE_SUMMARY_HW_OUT := results/processed/suite_fp32_summary_hw.csv
SUITE_PLOT_OUT := results/plots/roofline_fp32_primitives.svg

.PHONY: roofline-fp32 roofline-fp64 analyze-fp32 scaffold bench run-suite run-suite-hw summarize-suite summarize-suite-hw suite-fp32 suite-fp32-hw plot-roofline suite-fp32-hw-plot e2e-fp32

roofline-fp32:
	$(PYTHON) scripts/compute_roofline.py \
		--profile profiles/current_server.json \
		--precision fp32 \
		--output $(PROFILE_FP32_OUT)

roofline-fp64:
	$(PYTHON) scripts/compute_roofline.py \
		--profile profiles/current_server.json \
		--precision fp64 \
		--output $(PROFILE_FP64_OUT)

analyze-fp32: roofline-fp32
	$(PYTHON) scripts/analyze_primitive.py \
		--roofline $(PROFILE_FP32_OUT) \
		--input $(MEASUREMENTS) \
		--output $(ANALYSIS_FP32_OUT)

bench: $(BENCH_BIN)

$(BENCH_BIN): $(BENCH_SRC)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@

run-suite: bench
	$(PYTHON) scripts/run_suite.py \
		--config $(SUITE_CFG) \
		--output $(SUITE_RAW_OUT) \
		--bench $(BENCH_BIN) \
		--threads $(THREADS) \
		--seed $(SEED)

run-suite-hw: bench
	$(PYTHON) scripts/run_suite_hw_sim.py \
		--config $(SUITE_CFG) \
		--output $(SUITE_HW_SIM_OUT) \
		--bench $(BENCH_BIN) \
		--threads $(THREADS) \
		--seed $(SEED)

summarize-suite: roofline-fp32
	$(PYTHON) scripts/summarize_suite.py \
		--roofline $(PROFILE_FP32_OUT) \
		--raw $(SUITE_RAW_OUT) \
		--output $(SUITE_SUMMARY_OUT)

summarize-suite-hw: roofline-fp32
	$(PYTHON) scripts/summarize_suite.py \
		--roofline $(PROFILE_FP32_OUT) \
		--raw $(SUITE_RAW_OUT) \
		--hw-sim $(SUITE_HW_SIM_OUT) \
		--output $(SUITE_SUMMARY_HW_OUT)

suite-fp32: run-suite summarize-suite

suite-fp32-hw: run-suite run-suite-hw summarize-suite-hw

plot-roofline: suite-fp32-hw
	$(PYTHON) scripts/plot_roofline.py --roofline $(PROFILE_FP32_OUT) --summary $(SUITE_SUMMARY_HW_OUT) --output $(SUITE_PLOT_OUT) --title "FP32 Roofline + Primitive Points" $(if $(filter 1,$(SHOW_HW_IN_PLOT)),--show-hw,)

suite-fp32-hw-plot: plot-roofline

e2e-fp32:
	bash scripts/run_end_to_end_fp32.sh --threads $(THREADS) --seed $(SEED) $(if $(RUN_TAG),--tag $(RUN_TAG),) $(if $(filter 1,$(SHOW_HW_IN_PLOT)),--show-hw-in-plot,)

scaffold:
	@echo "usage: make scaffold NAME=<primitive_name>"
	@test -n "$(NAME)"
	./scripts/scaffold_primitive.sh "$(NAME)"
