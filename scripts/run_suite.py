#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path


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


def run_bench_json(
    bench: Path,
    workload: str,
    params: dict[str, str | int | float | bool],
    target_seconds: float,
    seed: int,
    threads: int,
) -> tuple[list[str], dict[str, str | int | float]]:
    with tempfile.NamedTemporaryFile(prefix="primitive_bench_", suffix=".json", delete=False) as f:
        json_path = Path(f.name)

    cmd = [
        str(bench),
        "--primitive",
        workload,
        "--target-seconds",
        str(target_seconds),
        "--seed",
        str(seed),
        "--json",
        str(json_path),
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

    try:
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        with json_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
    finally:
        if json_path.exists():
            json_path.unlink()

    if not isinstance(result, dict):
        raise RuntimeError("Benchmark JSON result must be an object.")
    return cmd, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run primitive benchmark suite.")
    parser.add_argument("--config", type=Path, required=True, help="CSV config for suite runs.")
    parser.add_argument("--output", type=Path, required=True, help="Raw output CSV.")
    parser.add_argument("--bench", type=str, default=None, help="Path to primitive_bench executable.")
    parser.add_argument("--threads", type=int, default=0, help="OMP threads override. 0 keeps row/default value.")
    parser.add_argument("--seed", type=int, default=42, help="Default RNG seed.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    bench = resolve_bench(args.bench, root_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.config.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Config CSV is empty.")

    out_rows: list[dict[str, str | int | float]] = []
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

        cmd, result = run_bench_json(
            bench=bench,
            workload=workload,
            params=params,
            target_seconds=target_seconds,
            seed=seed,
            threads=run_threads,
        )
        out = dict(result)
        out["run_cmd"] = " ".join(cmd)
        out_rows.append(out)

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in out_rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"rows: {len(out_rows)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
