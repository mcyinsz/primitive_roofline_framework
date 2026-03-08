#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def parse_positive_float(text: str | None) -> float | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if v <= 0.0:
        return None
    return v


def logspace(xmin: float, xmax: float, n: int = 256) -> list[float]:
    if xmin <= 0.0 or xmax <= 0.0 or xmax <= xmin:
        raise ValueError("logspace requires 0 < xmin < xmax")
    lo = math.log10(xmin)
    hi = math.log10(xmax)
    if n <= 1:
        return [10.0 ** lo]
    step = (hi - lo) / (n - 1)
    return [10.0 ** (lo + i * step) for i in range(n)]


def calc_ranges(
    points: list[dict[str, float | str | None]],
    peak: float,
    bw: float,
    ridge: float,
    zero_ai: float,
    right_ai: float,
    show_hw: bool,
) -> tuple[float, float, float, float]:
    ai_values = [float(p["ai"]) for p in points]
    x_min = min([zero_ai, min(ai_values) * 0.5, ridge * 0.2 if ridge > 0 else zero_ai])
    x_max = max([right_ai, max(ai_values) * 4.0, ridge * 4.0 if ridge > 0 else right_ai, 1.0])
    x_min = max(x_min, 1e-4)
    if x_max <= x_min:
        x_max = x_min * 10.0

    roofline_y = [min(peak, bw * x) for x in logspace(x_min, x_max, 200)]
    y_min = min(min(float(p["gflops"]) for p in points) * 0.6, max(1e-3, min(roofline_y) * 0.9))
    y_max = max(peak * 1.15, max(float(p["gflops"]) for p in points) * 1.5)
    if show_hw:
        hw_values = [float(p["hw_gflops"]) for p in points if p["hw_gflops"] is not None]
        if hw_values:
            y_max = max(y_max, max(hw_values) * 1.2)
    y_min = max(y_min, 1e-3)
    if y_max <= y_min:
        y_max = y_min * 10.0
    return x_min, x_max, y_min, y_max


def try_plot_matplotlib(
    output: Path,
    title: str,
    points: list[dict[str, float | str | None]],
    peak: float,
    bw: float,
    ridge: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    show_hw: bool,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x = logspace(x_min, x_max, 300)
    y = [min(peak, bw * xv) for xv in x]

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    ax.plot(x, y, color="#0c4a6e", linewidth=2.5, label="Roofline")
    ax.axvline(ridge, color="#475569", linestyle="--", linewidth=1.5, label=f"Ridge AI={ridge:.2f}")

    cmap = plt.get_cmap("tab10")
    for i, p in enumerate(points):
        color = cmap(i % 10)
        ax.scatter(
            float(p["ai"]),
            float(p["gflops"]),
            s=70,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
            label=str(p["primitive"]),
        )
        ax.annotate(
            str(p["primitive"]),
            (float(p["ai"]), float(p["gflops"])),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
            color="#0f172a",
        )
        if show_hw and p["hw_gflops"] is not None:
            ax.scatter(
                float(p["ai"]),
                float(p["hw_gflops"]),
                s=60,
                marker="x",
                color=color,
                linewidths=1.6,
                zorder=3,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.8)
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    dedup_h = []
    dedup_l = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        dedup_h.append(h)
        dedup_l.append(l)
    ax.legend(dedup_h, dedup_l, loc="best", fontsize=9)

    ax.set_xlim(left=x_min, right=x_max)
    ax.set_ylim(bottom=y_min, top=y_max)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved: {output}")
    print("backend: matplotlib")
    return True


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
    )


def tick_values(lo: float, hi: float) -> list[float]:
    a = int(math.floor(math.log10(lo)))
    b = int(math.ceil(math.log10(hi)))
    return [10.0 ** p for p in range(a, b + 1)]


def plot_svg_fallback(
    output: Path,
    title: str,
    points: list[dict[str, float | str | None]],
    peak: float,
    bw: float,
    ridge: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    show_hw: bool,
) -> Path:
    width = 1100
    height = 720
    left = 90
    right = 40
    top = 70
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    lx_min = math.log10(x_min)
    lx_max = math.log10(x_max)
    ly_min = math.log10(y_min)
    ly_max = math.log10(y_max)

    def tx(x: float) -> float:
        return left + (math.log10(x) - lx_min) / (lx_max - lx_min) * plot_w

    def ty(y: float) -> float:
        return top + (ly_max - math.log10(y)) / (ly_max - ly_min) * plot_h

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

    for xv in tick_values(x_min, x_max):
        if xv < x_min or xv > x_max:
            continue
        px = tx(xv)
        lines.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="{px:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="11" fill="#475569">{xv:g}</text>')

    for yv in tick_values(y_min, y_max):
        if yv < y_min or yv > y_max:
            continue
        py = ty(yv)
        lines.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{py + 4:.2f}" text-anchor="end" font-size="11" fill="#475569">{yv:g}</text>')

    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#0f172a" stroke-width="1.2"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#0f172a" stroke-width="1.2"/>')

    x_vals = logspace(x_min, x_max, 260)
    roof_pts = " ".join(f"{tx(x):.2f},{ty(min(peak, bw * x)):.2f}" for x in x_vals)
    lines.append(f'<polyline fill="none" stroke="#0c4a6e" stroke-width="2.5" points="{roof_pts}"/>')

    if ridge >= x_min and ridge <= x_max:
        rx = tx(ridge)
        lines.append(f'<line x1="{rx:.2f}" y1="{top}" x2="{rx:.2f}" y2="{top + plot_h}" stroke="#64748b" stroke-width="1.5" stroke-dasharray="6,4"/>')
        lines.append(f'<text x="{rx + 6:.2f}" y="{top + 16}" font-size="11" fill="#64748b">Ridge AI={ridge:.2f}</text>')

    palette = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4d7c0f", "#0f766e", "#9333ea"]
    for i, p in enumerate(points):
        color = palette[i % len(palette)]
        px = tx(float(p["ai"]))
        py = ty(float(p["gflops"]))
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.2" fill="{color}" stroke="black" stroke-width="0.6"/>')
        lines.append(f'<text x="{px + 6:.2f}" y="{py - 6:.2f}" font-size="11" fill="#0f172a">{svg_escape(str(p["primitive"]))}</text>')
        if show_hw and p["hw_gflops"] is not None:
            hy = ty(float(p["hw_gflops"]))
            lines.append(f'<line x1="{px - 4:.2f}" y1="{hy - 4:.2f}" x2="{px + 4:.2f}" y2="{hy + 4:.2f}" stroke="{color}" stroke-width="1.5"/>')
            lines.append(f'<line x1="{px - 4:.2f}" y1="{hy + 4:.2f}" x2="{px + 4:.2f}" y2="{hy - 4:.2f}" stroke="{color}" stroke-width="1.5"/>')

    lines.append(f'<text x="{width / 2:.2f}" y="34" text-anchor="middle" font-size="18" font-weight="600" fill="#0f172a">{svg_escape(title)}</text>')
    lines.append(f'<text x="{width / 2:.2f}" y="{height - 22}" text-anchor="middle" font-size="13" fill="#0f172a">Arithmetic Intensity (FLOP/Byte, log scale)</text>')
    lines.append(f'<text x="20" y="{height / 2:.2f}" transform="rotate(-90 20,{height / 2:.2f})" text-anchor="middle" font-size="13" fill="#0f172a">Performance (GFLOP/s, log scale)</text>')
    lines.append("</svg>")

    out = output
    if out.suffix.lower() != ".svg":
        out = out.with_suffix(".svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {out}")
    print("backend: svg-fallback")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot roofline line and primitive scatter points."
    )
    parser.add_argument("--roofline", type=Path, required=True, help="Roofline JSON.")
    parser.add_argument("--summary", type=Path, required=True, help="Suite summary CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output plot file path (PNG/SVG).")
    parser.add_argument("--title", type=str, default=None, help="Optional title.")
    parser.add_argument("--show-hw", action="store_true", help="Also plot optional hw_fp_gflops points.")
    args = parser.parse_args()

    with args.roofline.open("r", encoding="utf-8") as f:
        roof = json.load(f)

    peak = float(roof["peak_gflops"])
    bw = float(roof["memory_bandwidth_gbs"])
    ridge = float(roof["ridge_ai"])
    machine = str(roof.get("machine", "machine"))
    precision = str(roof.get("precision", "fp32"))
    zero_ai = float(roof.get("zero_ai", 0.01))
    right_ai = float(roof.get("right_ai", 100.0))

    with args.summary.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    points = []
    for r in rows:
        ai = parse_positive_float(r.get("ai"))
        gflops = parse_positive_float(r.get("achieved_gflops"))
        if ai is None or gflops is None:
            continue
        hw = None
        if args.show_hw:
            hw = parse_positive_float(r.get("hw_fp_gflops"))
        points.append({
            "primitive": r.get("primitive", ""),
            "ai": ai,
            "gflops": gflops,
            "hw_gflops": hw,
        })

    if not points:
        raise RuntimeError("No valid primitive points found in summary CSV.")

    x_min, x_max, y_min, y_max = calc_ranges(
        points=points,
        peak=peak,
        bw=bw,
        ridge=ridge,
        zero_ai=zero_ai,
        right_ai=right_ai,
        show_hw=args.show_hw,
    )

    default_title = f"{machine} {precision.upper()} Roofline with Primitive Points"
    title = args.title if args.title else default_title

    ok = try_plot_matplotlib(
        output=args.output,
        title=title,
        points=points,
        peak=peak,
        bw=bw,
        ridge=ridge,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        show_hw=args.show_hw,
    )
    if not ok:
        plot_svg_fallback(
            output=args.output,
            title=title,
            points=points,
            peak=peak,
            bw=bw,
            ridge=ridge,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            show_hw=args.show_hw,
        )

    print(f"points: {len(points)}")


if __name__ == "__main__":
    main()
