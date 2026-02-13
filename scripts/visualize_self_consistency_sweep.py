"""Visualize artifacts from scripts/modal_self_consistency_sweep.py.

Produces:
1) Required line chart: k vs execution_accuracy
2) Optional Pareto chart: execution_accuracy vs avg_total_tokens_per_example

By default, the script reads the latest directory under outputs/self_consistency_sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricPoint:
    k: int
    execution_accuracy: float | None
    avg_total_tokens_per_example: float | None
    duration_s: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize self-consistency sweep metrics."
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help=(
            "Path to a specific sweep directory (e.g. outputs/self_consistency_sweeps/20260212_213808). "
            "If omitted, the latest directory under --sweeps-root is used."
        ),
    )
    parser.add_argument(
        "--sweeps-root",
        default="outputs/self_consistency_sweeps",
        help="Root directory containing timestamped sweep folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write figures. Default: <sweep-dir>/visualizations",
    )
    parser.add_argument(
        "--skip-pareto",
        action="store_true",
        help="Skip the Pareto chart.",
    )
    parser.add_argument(
        "--pareto-x",
        choices=("avg_total_tokens_per_example", "duration_s"),
        default="avg_total_tokens_per_example",
        help="X-axis metric for Pareto chart (lower is better).",
    )
    return parser.parse_args()


def _resolve_sweep_dir(args: argparse.Namespace) -> Path:
    if args.sweep_dir:
        path = Path(args.sweep_dir)
        if not path.exists():
            raise SystemExit(f"Sweep directory not found: {path}")
        return path

    sweeps_root = Path(args.sweeps_root)
    if not sweeps_root.exists():
        raise SystemExit(f"Sweeps root not found: {sweeps_root}")

    candidates = [path for path in sweeps_root.iterdir() if path.is_dir()]
    if not candidates:
        raise SystemExit(f"No sweep directories found under: {sweeps_root}")
    return sorted(candidates)[-1]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _load_metric_rows(sweep_dir: Path) -> list[dict[str, Any]]:
    csv_path = sweep_dir / "metrics.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    json_path = sweep_dir / "metrics.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        raise SystemExit(f"Expected list in {json_path}, got: {type(payload).__name__}")

    raise SystemExit(
        f"No metrics file found in {sweep_dir}. Expected metrics.csv or metrics.json."
    )


def _parse_points(rows: list[dict[str, Any]]) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for row in rows:
        k = _to_int(row.get("k"))
        if k is None:
            continue
        points.append(
            MetricPoint(
                k=k,
                execution_accuracy=_to_float(row.get("execution_accuracy")),
                avg_total_tokens_per_example=_to_float(
                    row.get("avg_total_tokens_per_example")
                ),
                duration_s=_to_float(row.get("duration_s")),
            )
        )
    return points


def _pareto_frontier(points: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    """Return non-dominated points for objective: minimize x, maximize y."""
    frontier: list[tuple[int, float, float]] = []
    for candidate in points:
        _, cand_x, cand_y = candidate
        dominated = False
        for other in points:
            if other is candidate:
                continue
            _, oth_x, oth_y = other
            if (oth_x <= cand_x and oth_y >= cand_y) and (oth_x < cand_x or oth_y > cand_y):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: item[1])


def _scale(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max == in_min:
        return (out_min + out_max) / 2.0
    ratio = (value - in_min) / (in_max - in_min)
    return out_min + ratio * (out_max - out_min)


def _padded_domain(values: list[float], min_pad: float = 1.0, ratio: float = 0.08) -> tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        delta = max(min_pad, abs(maximum) * 0.05)
        return minimum - delta, maximum + delta
    delta = (maximum - minimum) * ratio
    return minimum - delta, maximum + delta


def _svg_document(width: int, height: int, elements: list[str]) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">',
            *elements,
            "</svg>",
        ]
    )


def _write_svg(path: Path, width: int, height: int, elements: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_svg_document(width, height, elements), encoding="utf-8")


def _plot_k_vs_exec_acc(points: list[MetricPoint], out_path: Path) -> None:
    filtered = [(point.k, point.execution_accuracy) for point in points if point.execution_accuracy is not None]
    if not filtered:
        raise SystemExit("No numeric execution_accuracy values found in metrics.")

    filtered.sort(key=lambda item: item[0])
    ks = [item[0] for item in filtered]

    width, height = 980, 620
    left, right = 90, width - 40
    top, bottom = 70, height - 90

    x_min = float(min(ks))
    x_max = float(max(ks))
    ys = [item[1] for item in filtered]
    y_min, y_max = float(min(ys)), float(max(ys))

    elements: list[str] = []
    elements.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    # Horizontal grid lines + y ticks.
    for y_tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y_tick = y_min + (y_max - y_min) * y_tick
        y_px = _scale(y_tick, y_min, y_max, bottom, top)
        elements.append(
            f'<line x1="{left}" y1="{y_px:.2f}" x2="{right}" y2="{y_px:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 12}" y="{y_px + 4:.2f}" text-anchor="end" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#374151">'
            f"{y_tick:.2f}</text>"
        )

    # Vertical ticks by k.
    for k, _ in filtered:
        x_px = _scale(float(k), x_min, x_max, left, right)
        elements.append(
            f'<line x1="{x_px:.2f}" y1="{bottom}" x2="{x_px:.2f}" y2="{bottom + 6}" '
            'stroke="#6b7280" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x_px:.2f}" y="{bottom + 24}" text-anchor="middle" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#374151">'
            f"{k}</text>"
        )

    # Axes.
    elements.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#111827" stroke-width="1.5"/>'
    )
    elements.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111827" stroke-width="1.5"/>'
    )

    poly_points: list[str] = []
    for k, exec_acc in filtered:
        x_px = _scale(float(k), x_min, x_max, left, right)
        y_px = _scale(float(exec_acc), y_min, y_max, bottom, top)
        poly_points.append(f"{x_px:.2f},{y_px:.2f}")
    elements.append(
        f'<polyline points="{" ".join(poly_points)}" fill="none" stroke="#2563eb" stroke-width="3"/>'
    )

    for k, exec_acc in filtered:
        x_px = _scale(float(k), x_min, x_max, left, right)
        y_px = _scale(float(exec_acc), y_min, y_max, bottom, top)
        elements.append(f'<circle cx="{x_px:.2f}" cy="{y_px:.2f}" r="5" fill="#1d4ed8"/>')
        elements.append(
            f'<text x="{x_px:.2f}" y="{y_px - 10:.2f}" text-anchor="middle" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#1e3a8a">'
            f"k={k}</text>"
        )

    # Titles and axis labels.
    elements.append(
        f'<text x="{width / 2:.2f}" y="36" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#111827">'
        "Self-Consistency Sweep: k vs Execution Accuracy</text>"
    )
    elements.append(
        f'<text x="{width / 2:.2f}" y="{height - 32}" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#111827">'
        "Self-consistency samples (k)</text>"
    )
    elements.append(
        f'<text x="24" y="{height / 2:.2f}" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#111827" '
        f'transform="rotate(-90 24 {height / 2:.2f})">'
        "Execution accuracy</text>"
    )

    _write_svg(out_path, width, height, elements)


def _plot_pareto(
    points: list[MetricPoint],
    out_path: Path,
    x_metric: str,
) -> bool:
    tuples: list[tuple[int, float, float]] = []
    for point in points:
        x_value = (
            point.avg_total_tokens_per_example
            if x_metric == "avg_total_tokens_per_example"
            else point.duration_s
        )
        if x_value is None or point.execution_accuracy is None:
            continue
        tuples.append((point.k, x_value, point.execution_accuracy))

    if len(tuples) < 2:
        return False

    frontier = _pareto_frontier(tuples)

    width, height = 980, 620
    left, right = 90, width - 40
    top, bottom = 70, height - 90

    x_values = [item[1] for item in tuples]
    x_min, x_max = _padded_domain(x_values, min_pad=1.0)
    y_min, y_max = 0.0, 1.0

    elements: list[str] = []
    elements.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    # Horizontal grid + y ticks.
    for y_tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y_px = _scale(y_tick, y_min, y_max, bottom, top)
        elements.append(
            f'<line x1="{left}" y1="{y_px:.2f}" x2="{right}" y2="{y_px:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 12}" y="{y_px + 4:.2f}" text-anchor="end" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#374151">'
            f"{y_tick:.2f}</text>"
        )

    # X ticks with 5 buckets.
    x_tick_values = [x_min + (x_max - x_min) * (index / 4.0) for index in range(5)]
    for tick in x_tick_values:
        x_px = _scale(tick, x_min, x_max, left, right)
        elements.append(
            f'<line x1="{x_px:.2f}" y1="{bottom}" x2="{x_px:.2f}" y2="{bottom + 6}" '
            'stroke="#6b7280" stroke-width="1"/>'
        )
        tick_label = f"{tick:,.0f}" if x_metric == "avg_total_tokens_per_example" else f"{tick:.0f}"
        elements.append(
            f'<text x="{x_px:.2f}" y="{bottom + 24}" text-anchor="middle" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#374151">'
            f"{tick_label}</text>"
        )

    elements.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#111827" stroke-width="1.5"/>'
    )
    elements.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111827" stroke-width="1.5"/>'
    )

    palette = [
        "#2563eb",
        "#16a34a",
        "#dc2626",
        "#d97706",
        "#7c3aed",
        "#0891b2",
        "#db2777",
        "#4f46e5",
    ]
    color_by_k: dict[int, str] = {}
    for idx, k in enumerate(sorted({item[0] for item in tuples})):
        color_by_k[k] = palette[idx % len(palette)]

    for k, x_value, y_value in tuples:
        x_px = _scale(x_value, x_min, x_max, left, right)
        y_px = _scale(y_value, y_min, y_max, bottom, top)
        color = color_by_k[k]
        elements.append(
            f'<circle cx="{x_px:.2f}" cy="{y_px:.2f}" r="6" fill="{color}" stroke="#111827" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x_px:.2f}" y="{y_px - 10:.2f}" text-anchor="middle" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#111827">'
            f"k={k}</text>"
        )

    if len(frontier) >= 2:
        frontier_points = [
            f'{_scale(x_value, x_min, x_max, left, right):.2f},{_scale(y_value, y_min, y_max, bottom, top):.2f}'
            for _, x_value, y_value in frontier
        ]
        elements.append(
            f'<polyline points="{" ".join(frontier_points)}" fill="none" '
            'stroke="#b91c1c" stroke-width="2.5" stroke-dasharray="8 6"/>'
        )
        elements.append(
            f'<text x="{right - 160}" y="{top + 18}" text-anchor="start" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#b91c1c">'
            "Pareto frontier</text>"
        )

    x_label = (
        "Avg total tokens per example (lower is better)"
        if x_metric == "avg_total_tokens_per_example"
        else "Duration (seconds, lower is better)"
    )
    elements.append(
        f'<text x="{width / 2:.2f}" y="36" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#111827">'
        "Pareto View: Cost vs Execution Accuracy</text>"
    )
    elements.append(
        f'<text x="{width / 2:.2f}" y="{height - 32}" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#111827">'
        f"{escape(x_label)}</text>"
    )
    elements.append(
        f'<text x="24" y="{height / 2:.2f}" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#111827" '
        f'transform="rotate(-90 24 {height / 2:.2f})">'
        "Execution accuracy (higher is better)</text>"
    )

    _write_svg(out_path, width, height, elements)
    return True


def main() -> int:
    args = _parse_args()
    sweep_dir = _resolve_sweep_dir(args)
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "visualizations"

    rows = _load_metric_rows(sweep_dir)
    points = _parse_points(rows)
    if not points:
        raise SystemExit(f"No valid metric points found in {sweep_dir}.")

    k_plot_path = output_dir / "k_vs_exec_acc.svg"
    _plot_k_vs_exec_acc(points, k_plot_path)
    print(f"Wrote: {k_plot_path}")

    if not args.skip_pareto:
        pareto_name = (
            "pareto_exec_acc_vs_tokens.svg"
            if args.pareto_x == "avg_total_tokens_per_example"
            else "pareto_exec_acc_vs_duration.svg"
        )
        pareto_path = output_dir / pareto_name
        wrote = _plot_pareto(points, pareto_path, x_metric=args.pareto_x)
        if wrote:
            print(f"Wrote: {pareto_path}")
        else:
            print("Skipped Pareto plot (need at least 2 rows with numeric x metric and execution_accuracy).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
