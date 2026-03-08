#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_profile(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_roofline(profile: dict, precision: str, zero_ai: float, right_ai: float) -> dict:
    cpu = profile["cpu"]
    mem = profile["memory"]

    cores_total = cpu.get("cores_total", cpu["sockets"] * cpu["cores_per_socket"])
    freq_ghz = cpu["frequency_ghz"]
    flops_per_cycle = cpu[f"flops_per_cycle_{precision}"]

    peak_flops = cores_total * freq_ghz * 1e9 * flops_per_cycle
    bandwidth_bps = mem["channels_total"] * mem["mt_s"] * 1e6 * mem.get("bytes_per_transfer", 8)

    ridge_ai = peak_flops / bandwidth_bps

    zero_gflops = (zero_ai * bandwidth_bps) / 1e9
    right_gflops = min((right_ai * bandwidth_bps) / 1e9, peak_flops / 1e9)

    return {
        "machine": profile["name"],
        "precision": precision,
        "cores_total": cores_total,
        "frequency_ghz": freq_ghz,
        "flops_per_cycle": flops_per_cycle,
        "peak_flops": peak_flops,
        "peak_gflops": peak_flops / 1e9,
        "memory_bandwidth_bps": bandwidth_bps,
        "memory_bandwidth_gbs": bandwidth_bps / 1e9,
        "zero_ai": zero_ai,
        "ridge_ai": ridge_ai,
        "right_ai": right_ai,
        "zero_gflops": zero_gflops,
        "ridge_gflops": peak_flops / 1e9,
        "right_gflops": right_gflops
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute roofline constants from a machine profile.")
    parser.add_argument("--profile", type=Path, required=True, help="Path to machine profile JSON.")
    parser.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    parser.add_argument("--zero-ai", type=float, default=0.01)
    parser.add_argument("--right-ai", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    result = compute_roofline(profile, args.precision, args.zero_ai, args.right_ai)

    print(f"machine: {result['machine']}")
    print(f"precision: {result['precision']}")
    print(f"peak_gflops: {result['peak_gflops']:.4f}")
    print(f"memory_bandwidth_gbs: {result['memory_bandwidth_gbs']:.4f}")
    print(f"ridge_ai: {result['ridge_ai']:.6f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

