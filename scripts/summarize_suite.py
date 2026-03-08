#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_int_field(row: dict[str, str], key: str) -> int:
    value = row.get(key, "").strip()
    if not value:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def parse_float_field(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def row_key(row: dict[str, str]) -> tuple[str, int, int, int, int]:
    return (
        row.get("primitive", "").strip(),
        parse_int_field(row, "dim"),
        parse_int_field(row, "scale"),
        parse_int_field(row, "db_vectors"),
        parse_int_field(row, "threads"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize primitive raw benchmark against roofline model.")
    parser.add_argument("--roofline", type=Path, required=True, help="Roofline JSON.")
    parser.add_argument("--raw", type=Path, required=True, help="Raw benchmark CSV from run_suite.py.")
    parser.add_argument("--hw-sim", type=Path, default=None, help="Optional hardware simulation CSV from run_suite_hw_sim.py.")
    parser.add_argument("--output", type=Path, required=True, help="Output summary CSV.")
    args = parser.parse_args()

    with args.roofline.open("r", encoding="utf-8") as f:
        roofline = json.load(f)
    peak_gflops = float(roofline["peak_gflops"])
    bw_gbs = float(roofline["memory_bandwidth_gbs"])
    ridge_ai = float(roofline["ridge_ai"])

    with args.raw.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    hw_map: dict[tuple[str, int, int, int, int], dict[str, str]] = {}
    if args.hw_sim is not None:
        with args.hw_sim.open("r", encoding="utf-8") as f:
            for h in csv.DictReader(f):
                hw_map[row_key(h)] = h

    out_rows = []
    for row in rows:
        ai = float(row["ai"])
        achieved = float(row["achieved_gflops"])
        cap = min(peak_gflops, ai * bw_gbs)
        eff = 0.0 if cap <= 0 else achieved / cap * 100.0
        bound = "memory" if ai < ridge_ai else "compute"

        out = dict(row)
        out["roofline_cap_gflops"] = f"{cap:.6f}"
        out["roofline_bound"] = bound
        out["efficiency_pct_of_cap"] = f"{eff:.3f}"
        out["headroom_gflops"] = f"{(cap - achieved):.6f}"

        hw = hw_map.get(row_key(row))
        if hw is not None:
            out["hw_fp_gflops"] = hw.get("hw_fp_gflops", "")
            out["hw_fp_ops"] = hw.get("hw_fp_ops", "")
            out["hw_cycles"] = hw.get("hw_cycles", "")
            out["hw_instructions"] = hw.get("hw_instructions", "")
            out["hw_ipc"] = hw.get("hw_ipc", "")
            out["hw_dram_bw_gbs"] = hw.get("hw_dram_bw_gbs", "")
            out["hw_dram_total_mib"] = hw.get("hw_dram_total_mib", "")

            hw_fp_gflops = parse_float_field(hw, "hw_fp_gflops")
            hw_bw = parse_float_field(hw, "hw_dram_bw_gbs")
            if hw_bw is not None and hw_bw > 0.0:
                hw_cap = min(peak_gflops, ai * hw_bw)
                hw_eff = 0.0 if hw_cap <= 0.0 else (0.0 if hw_fp_gflops is None else hw_fp_gflops / hw_cap * 100.0)
                hw_bound = "memory" if ai < (peak_gflops / hw_bw) else "compute"
                hw_headroom = (0.0 if hw_fp_gflops is None else (hw_cap - hw_fp_gflops))
                out["hw_roofline_cap_gflops"] = f"{hw_cap:.6f}"
                out["hw_roofline_bound"] = hw_bound
                out["hw_efficiency_pct_of_cap"] = f"{hw_eff:.3f}"
                out["hw_headroom_gflops"] = f"{hw_headroom:.6f}"
            else:
                out["hw_roofline_cap_gflops"] = ""
                out["hw_roofline_bound"] = ""
                out["hw_efficiency_pct_of_cap"] = ""
                out["hw_headroom_gflops"] = ""
        else:
            out["hw_fp_gflops"] = ""
            out["hw_fp_ops"] = ""
            out["hw_cycles"] = ""
            out["hw_instructions"] = ""
            out["hw_ipc"] = ""
            out["hw_dram_bw_gbs"] = ""
            out["hw_dram_total_mib"] = ""
            out["hw_roofline_cap_gflops"] = ""
            out["hw_roofline_bound"] = ""
            out["hw_efficiency_pct_of_cap"] = ""
            out["hw_headroom_gflops"] = ""

        out_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(out_rows[0].keys()) if out_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"rows: {len(out_rows)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
