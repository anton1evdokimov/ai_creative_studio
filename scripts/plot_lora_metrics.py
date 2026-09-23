#!/usr/bin/env python3
"""Plot LoRA train metrics.csv (loss, CLIP-T/I, DINO-I, aesthetic).

    python scripts/plot_lora_metrics.py --csv lora/xyz_hoodie/metrics.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "step",
    "loss",
    "clip_t",
    "clip_t_raw",
    "clip_i",
    "clip_i_raw",
    "dino_i",
    "dino_i_raw",
    "aesthetic",
    "vlm_overall",
    "ocr_hit",
]


def _f(v) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_rows(path: Path) -> list[dict]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise SystemExit(f"Empty file: {path}")
    parsed = list(csv.reader(raw_lines))
    header, body = parsed[0], parsed[1:]
    out = []
    for row in body:
        if not row or not str(row[0]).strip().isdigit():
            continue
        # New schema written after header drift: 14 columns, old 9-col header
        if len(row) >= 9 and "clip_t_raw" not in header:
            if len(row) >= 14:
                keys = [
                    "step", "loss", "clip_t", "clip_t_raw", "clip_i", "clip_i_raw",
                    "dino_i", "dino_i_raw", "aesthetic", "vlm_overall", "ocr_hit",
                    "ocr_text", "clip_error", "prompt",
                ]
            else:
                keys = header
        else:
            keys = header
        rec = {k: (row[i] if i < len(row) else "") for i, k in enumerate(keys)}
        rec["step"] = int(float(rec["step"]))
        for k in FIELDS:
            if k == "step":
                continue
            rec[k] = _f(rec.get(k))
        out.append(rec)
    if not out:
        raise SystemExit(f"No numeric rows in {path}")
    out.sort(key=lambda r: r["step"])
    return out


def _series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in rows:
        if r.get(key) is None:
            continue
        xs.append(r["step"])
        ys.append(r[key])
    return xs, ys


def plot(rows: list[dict], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("pip install matplotlib") from exc

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax = axes[0]
    x, y = _series(rows, "loss")
    ax.plot(x, y, "o-", color="#c0392b", label="train MSE")
    ax.set_ylabel("loss")
    ax.set_title("LoRA train metrics")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    for key, color, label in (
        ("clip_t", "#2980b9", "CLIP-T (prompt)"),
        ("clip_i", "#27ae60", "CLIP-I (identity)"),
        ("dino_i", "#8e44ad", "DINO-I (identity)"),
        ("aesthetic", "#e67e22", "LAION aesthetic"),
        ("vlm_overall", "#16a085", "VLM overall"),
        ("ocr_hit", "#7f8c8d", "OCR hit"),
    ):
        x, y = _series(rows, key)
        if x:
            ax.plot(x, y, "o-", color=color, label=label)
    ax.set_ylabel("score 0–1")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[2]
    for key, color, label in (
        ("clip_t_raw", "#2980b9", "CLIP-T cosine"),
        ("clip_i_raw", "#27ae60", "CLIP-I cosine"),
        ("dino_i_raw", "#8e44ad", "DINO cosine"),
    ):
        x, y = _series(rows, key)
        if x:
            ax.plot(x, y, "o-", color=color, label=label)
    ax.set_ylabel("raw cosine")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> int:
    p = argparse.ArgumentParser(description="Plot LoRA metrics.csv")
    p.add_argument("--csv", default="lora/xyz_hoodie/metrics.csv")
    p.add_argument("--out", default="", help="PNG path (default: next to csv)")
    args = p.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        old = csv_path.with_suffix(csv_path.suffix + ".old")
        if old.exists():
            print(f"{csv_path} missing, using {old}")
            csv_path = old
        else:
            raise SystemExit(f"Not found: {csv_path}")
    rows = load_rows(csv_path)
    out = Path(args.out) if args.out else csv_path.with_suffix(".png")
    plot(rows, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
