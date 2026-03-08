#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze primitive measurements against a roofline model.")
    parser.add_argument("--roofline", type=Path, required=True, help="Roofline JSON from compute_roofline.py.")
    parser.add_argument("--input", type=Path, required=True, help="Measurement CSV with ai and achieved_gflops fields.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV with roofline columns appended.")
    args = parser.parse_args()

    with args.roofline.open("r", encoding="utf-8") as f:
        roof = json.load(f)
    peak_gflops = float(roof["peak_gflops"])
    bw_gbs = float(roof["memory_bandwidth_gbs"])
    ridge_ai = float(roof["ridge_ai"])

    with args.input.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Input CSV is empty.")

    out_rows: list[dict[str, str]] = []
    for row in rows:
        ai = float(row["ai"])
        achieved = float(row["achieved_gflops"])
        cap = min(peak_gflops, ai * bw_gbs)
        bound = "memory" if ai < ridge_ai else "compute"
        eff = 0.0 if cap <= 0.0 else achieved / cap * 100.0

        out = dict(row)
        out["roofline_cap_gflops"] = f"{cap:.6f}"
        out["roofline_bound"] = bound
        out["efficiency_pct_of_cap"] = f"{eff:.3f}"
        out["headroom_gflops"] = f"{(cap - achieved):.6f}"
        out_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(out_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"rows: {len(out_rows)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
