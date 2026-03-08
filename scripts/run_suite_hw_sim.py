#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import tempfile
from pathlib import Path


FP_EVENTS_A = [
    "fp_arith_inst_retired.scalar_single",
    "fp_arith_inst_retired.scalar_double",
    "fp_arith_inst_retired.128b_packed_single",
    "fp_arith_inst_retired.128b_packed_double",
]

FP_EVENTS_B = [
    "fp_arith_inst_retired.256b_packed_single",
    "fp_arith_inst_retired.256b_packed_double",
    "fp_arith_inst_retired.512b_packed_single",
    "fp_arith_inst_retired.512b_packed_double",
]

CORE_META_EVENTS = ["cycles", "instructions"]

FP_WEIGHTS = {
    "fp_arith_inst_retired.scalar_single": 1.0,
    "fp_arith_inst_retired.scalar_double": 1.0,
    "fp_arith_inst_retired.128b_packed_single": 4.0,
    "fp_arith_inst_retired.128b_packed_double": 2.0,
    "fp_arith_inst_retired.256b_packed_single": 8.0,
    "fp_arith_inst_retired.256b_packed_double": 4.0,
    "fp_arith_inst_retired.512b_packed_single": 16.0,
    "fp_arith_inst_retired.512b_packed_double": 8.0,
}


def resolve_bench(path_arg: str | None, root_dir: Path) -> Path:
    if path_arg:
        p = Path(path_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Benchmark executable not found: {p}")

    candidates = [
        root_dir / "bin" / "primitive_bench",
        root_dir / ".." / "build" / "PrimitiveBench",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find primitive bench executable. Checked: {candidates}")


def parse_float(text: str | None, default: float) -> float:
    if text is None:
        return default
    s = text.strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(f"Cannot parse float value: {text!r}") from exc


def parse_int(text: str | None, default: int) -> int:
    if text is None:
        return default
    s = text.strip()
    if not s:
        return default
    try:
        return int(float(s))
    except ValueError as exc:
        raise ValueError(f"Cannot parse integer value: {text!r}") from exc


def parse_scalar(text: str) -> str | int | float | bool:
    s = text.strip()
    if not s:
        return s
    t = s.lower()
    if t == "true":
        return True
    if t == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_params(row: dict[str, str]) -> dict[str, str | int | float | bool]:
    params_raw = row.get("params_json", "").strip()
    if params_raw:
        parsed = json.loads(params_raw)
        if not isinstance(parsed, dict):
            raise ValueError("params_json must decode to a JSON object.")
        return parsed

    # Backward-compat: old suite schema with explicit columns.
    params: dict[str, str | int | float | bool] = {}
    if row.get("dim", "").strip():
        params["dim"] = parse_int(row.get("dim"), 0)
    if row.get("scale", "").strip():
        params["scale"] = parse_int(row.get("scale"), 0)
    if row.get("db_vectors", "").strip():
        params["db_vectors"] = parse_int(row.get("db_vectors"), 0)

    reserved = {
        "primitive",
        "workload",
        "params_json",
        "target_seconds",
        "seed",
        "threads",
        "dim",
        "scale",
        "db_vectors",
    }
    for k, v in row.items():
        if k in reserved:
            continue
        if not str(v).strip():
            continue
        params[k] = parse_scalar(str(v))
    return params


def choose_workload(row: dict[str, str], params: dict[str, str | int | float | bool]) -> str:
    w = row.get("workload", "").strip()
    if w:
        return w
    p = row.get("primitive", "").strip()
    if p:
        return p
    pv = params.get("workload")
    if isinstance(pv, str) and pv.strip():
        return pv.strip()
    pv = params.get("primitive")
    if isinstance(pv, str) and pv.strip():
        return pv.strip()
    raise ValueError("Each row must provide workload or primitive.")


def build_base_cmd(
    bench: Path,
    workload: str,
    params: dict[str, str | int | float | bool],
    target_seconds: float,
    seed: int,
    threads: int,
) -> list[str]:
    cmd = [
        str(bench),
        "--primitive",
        workload,
        "--target-seconds",
        str(target_seconds),
        "--seed",
        str(seed),
    ]
    if threads > 0:
        cmd.extend(["--threads", str(threads)])

    skip = {
        "workload",
        "primitive",
        "target_seconds",
        "target-seconds",
        "seed",
        "threads",
    }
    int_param_keys = {
        "dim",
        "scale",
        "db_vectors",
        "fixed_repetitions",
    }
    for key in sorted(params.keys()):
        if key in skip:
            continue
        value = params[key]
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if value is None:
            continue
        if key in int_param_keys:
            cmd.extend([flag, str(int(float(value)))])
        else:
            cmd.extend([flag, str(value)])
    return cmd


def parse_perf_value(value_text: str) -> float | None:
    t = value_text.strip().replace(" ", "")
    if not t:
        return None
    if t.startswith("<") and t.endswith(">"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def parse_perf_csv(stderr_text: str) -> dict[str, dict[str, float | str | None]]:
    out: dict[str, dict[str, float | str | None]] = {}
    for line in stderr_text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        value_txt = parts[0].strip()
        unit = parts[1].strip()
        event = parts[2].strip()
        if not event:
            continue
        out[event] = {
            "value": parse_perf_value(value_txt),
            "unit": unit,
            "raw_value": value_txt,
        }
    return out


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_bench_json(bench_cmd: list[str]) -> dict[str, str | int | float]:
    with tempfile.NamedTemporaryFile(prefix="primitive_bench_", suffix=".json", delete=False) as f:
        json_path = Path(f.name)
    cmd = [*bench_cmd, "--json", str(json_path)]
    try:
        proc = run_cmd(cmd)
        if proc.stdout:
            print(proc.stdout.strip())
        if proc.stderr:
            print(proc.stderr.strip())
        with json_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
    finally:
        if json_path.exists():
            json_path.unlink()
    if not isinstance(result, dict):
        raise RuntimeError("Benchmark JSON result must be an object.")
    return result


def run_perf(
    perf_bin: str,
    bench_cmd: list[str],
    events: list[str],
) -> tuple[dict[str, str | int | float], dict[str, dict[str, float | str | None]]]:
    with tempfile.NamedTemporaryFile(prefix="primitive_bench_", suffix=".json", delete=False) as f:
        json_path = Path(f.name)
    perf_cmd = [perf_bin, "stat", "-x,", "--no-big-num", "-e", ",".join(events), "--", *bench_cmd, "--json", str(json_path)]
    try:
        proc = run_cmd(perf_cmd)
        bench_result = {}
        with json_path.open("r", encoding="utf-8") as f:
            bench_result = json.load(f)
        perf_stats = parse_perf_csv(proc.stderr)
    finally:
        if json_path.exists():
            json_path.unlink()

    if not isinstance(bench_result, dict):
        raise RuntimeError("Benchmark JSON result must be an object.")
    return bench_result, perf_stats


def safe_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v))
    except ValueError:
        return None


def counter(stats: dict[str, dict[str, float | str | None]], name: str) -> float | None:
    if name not in stats:
        return None
    value = stats[name].get("value")
    if isinstance(value, (float, int)):
        return float(value)
    return None


def bytes_from_counter(value: float | None, unit: str, event: str) -> float:
    if value is None:
        return 0.0
    u = unit.strip().lower()
    if u == "mib":
        return value * 1024.0 * 1024.0
    if u == "gib":
        return value * 1024.0 * 1024.0 * 1024.0
    if u == "kib":
        return value * 1024.0
    if u in ("b", "bytes"):
        return value
    # Fallback if kernel exposes raw CAS counts without unit conversion.
    if "cas_count" in event:
        return value * 64.0
    return value


def get_perf_list_text(perf_bin: str) -> str:
    proc = run_cmd([perf_bin, "list"])
    return proc.stdout + "\n" + proc.stderr


def detect_imc_events(perf_list_text: str) -> list[str]:
    found = re.findall(r"(uncore_imc_\d+/cas_count_(?:read|write)/)", perf_list_text)
    return sorted(set(found), key=imc_sort_key)


def imc_sort_key(event: str) -> tuple[int, int]:
    m = re.match(r"uncore_imc_(\d+)/cas_count_(read|write)/", event)
    if not m:
        return (1 << 30, 1 << 30)
    channel = int(m.group(1))
    rw = 0 if m.group(2) == "read" else 1
    return (channel, rw)


def format_float(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.6f}"


def filter_supported_events(events: list[str], perf_list_text: str) -> list[str]:
    out: list[str] = []
    for ev in events:
        if re.search(rf"\b{re.escape(ev)}\b", perf_list_text):
            out.append(ev)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run primitive suite with perf-based hardware simulation metrics.")
    parser.add_argument("--config", type=Path, required=True, help="CSV config for suite runs.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV for hardware simulation metrics.")
    parser.add_argument("--bench", type=str, default=None, help="Path to primitive_bench executable.")
    parser.add_argument("--threads", type=int, default=0, help="OMP threads override. 0 keeps row/default value.")
    parser.add_argument("--seed", type=int, default=42, help="Default RNG seed.")
    parser.add_argument("--perf-bin", type=str, default="perf", help="perf executable path.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    bench = resolve_bench(args.bench, root_dir)
    perf_list_text = get_perf_list_text(args.perf_bin)
    imc_events = detect_imc_events(perf_list_text)
    fp_events_a = filter_supported_events(FP_EVENTS_A, perf_list_text)
    fp_events_b = filter_supported_events(FP_EVENTS_B, perf_list_text)
    if not fp_events_a and not fp_events_b:
        raise RuntimeError("No supported fp_arith_inst_retired events found in `perf list`.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    with args.config.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Config CSV is empty.")

    out_rows: list[dict[str, str]] = []
    for row in rows:
        params = parse_params(row)
        workload = choose_workload(row, params)
        target_seconds = parse_float(row.get("target_seconds"), default=1.0)
        if "target_seconds" in params:
            target_seconds = float(params["target_seconds"])

        seed = parse_int(row.get("seed"), default=args.seed)
        if "seed" in params:
            seed = int(float(params["seed"]))

        row_threads = parse_int(row.get("threads"), default=0)
        if "threads" in params:
            row_threads = int(float(params["threads"]))
        run_threads = args.threads if args.threads > 0 else row_threads

        base_cmd = build_base_cmd(
            bench=bench,
            workload=workload,
            params=params,
            target_seconds=target_seconds,
            seed=seed,
            threads=run_threads,
        )

        print("baseline:", " ".join(base_cmd))
        baseline = run_bench_json(base_cmd)
        reps = int(float(baseline.get("repetitions", 0)))
        if reps <= 0:
            raise RuntimeError(f"Cannot parse repetitions from baseline output for workload={workload}")

        fixed_cmd = [*base_cmd, "--fixed-repetitions", str(reps)]

        print("perf-core:", " ".join(fixed_cmd))
        core_run, meta_stats = run_perf(args.perf_bin, fixed_cmd, CORE_META_EVENTS)

        fp_elapsed_values: list[float] = []
        fp_ops = 0.0
        if fp_events_a:
            print("perf-fp-a:", " ".join(fixed_cmd))
            fp_run_a, fp_stats_a = run_perf(args.perf_bin, fixed_cmd, fp_events_a)
            elapsed_a = safe_float(fp_run_a.get("elapsed_sec"))
            if elapsed_a and elapsed_a > 0.0:
                fp_elapsed_values.append(elapsed_a)
            for ev, weight in FP_WEIGHTS.items():
                c = counter(fp_stats_a, ev)
                if c is not None:
                    fp_ops += c * weight
        if fp_events_b:
            print("perf-fp-b:", " ".join(fixed_cmd))
            fp_run_b, fp_stats_b = run_perf(args.perf_bin, fixed_cmd, fp_events_b)
            elapsed_b = safe_float(fp_run_b.get("elapsed_sec"))
            if elapsed_b and elapsed_b > 0.0:
                fp_elapsed_values.append(elapsed_b)
            for ev, weight in FP_WEIGHTS.items():
                c = counter(fp_stats_b, ev)
                if c is not None:
                    fp_ops += c * weight

        mem_stats: dict[str, dict[str, float | str | None]] = {}
        mem_elapsed = None
        if imc_events:
            print("perf-mem :", " ".join(fixed_cmd))
            mem_run, mem_stats = run_perf(args.perf_bin, fixed_cmd, imc_events)
            mem_elapsed = safe_float(mem_run.get("elapsed_sec"))

        fp_elapsed = None
        if fp_elapsed_values:
            fp_elapsed = sum(fp_elapsed_values) / len(fp_elapsed_values)
        hw_fp_gflops = None
        if fp_elapsed is not None and fp_elapsed > 0.0:
            hw_fp_gflops = fp_ops / fp_elapsed / 1e9

        core_elapsed = safe_float(core_run.get("elapsed_sec"))
        cycles = counter(meta_stats, "cycles")
        instructions = counter(meta_stats, "instructions")
        ipc = None
        if cycles and cycles > 0.0 and instructions is not None:
            ipc = instructions / cycles

        read_bytes = 0.0
        write_bytes = 0.0
        for ev in imc_events:
            st = mem_stats.get(ev, {})
            value = st.get("value")
            unit = str(st.get("unit", ""))
            if not isinstance(value, (int, float)):
                continue
            b = bytes_from_counter(float(value), unit, ev)
            if ev.endswith("cas_count_read/"):
                read_bytes += b
            elif ev.endswith("cas_count_write/"):
                write_bytes += b
        total_bytes = read_bytes + write_bytes

        hw_bw_gbs = None
        if mem_elapsed and mem_elapsed > 0.0 and total_bytes > 0.0:
            hw_bw_gbs = total_bytes / mem_elapsed / 1e9

        out_rows.append({
            "primitive": str(baseline.get("primitive", workload)),
            "dim": str(int(float(baseline.get("dim", 1)))),
            "scale": str(int(float(baseline.get("scale", 0)))),
            "db_vectors": str(int(float(baseline.get("db_vectors", 0)))),
            "target_seconds": str(baseline.get("target_seconds", target_seconds)),
            "threads": str(int(float(baseline.get("threads", run_threads if run_threads > 0 else 0)))),
            "seed": str(int(float(baseline.get("seed", seed)))),
            "repetitions": str(reps),
            "elapsed_sec": format_float(safe_float(baseline.get("elapsed_sec"))),
            "model_ai": format_float(safe_float(baseline.get("ai"))),
            "model_achieved_gflops": format_float(safe_float(baseline.get("achieved_gflops"))),
            "model_achieved_gbs": format_float(safe_float(baseline.get("achieved_gbs"))),
            "hw_fp_ops": format_float(fp_ops),
            "hw_fp_gflops": format_float(hw_fp_gflops),
            "hw_fp_elapsed_sec": format_float(fp_elapsed),
            "hw_cycles": format_float(cycles),
            "hw_instructions": format_float(instructions),
            "hw_ipc": format_float(ipc),
            "hw_core_elapsed_sec": format_float(core_elapsed),
            "hw_dram_read_bytes": format_float(read_bytes),
            "hw_dram_write_bytes": format_float(write_bytes),
            "hw_dram_total_bytes": format_float(total_bytes),
            "hw_dram_bw_gbs": format_float(hw_bw_gbs),
            "hw_mem_elapsed_sec": format_float(mem_elapsed),
            "hw_dram_read_mib": format_float(read_bytes / (1024.0 * 1024.0)),
            "hw_dram_write_mib": format_float(write_bytes / (1024.0 * 1024.0)),
            "hw_dram_total_mib": format_float(total_bytes / (1024.0 * 1024.0)),
            "hw_events_fp_a": ";".join(fp_events_a),
            "hw_events_fp_b": ";".join(fp_events_b),
            "hw_events_mem": ";".join(imc_events),
            "run_cmd_fixed": " ".join(fixed_cmd),
        })

    with args.output.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(out_rows[0].keys()) if out_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"rows: {len(out_rows)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
