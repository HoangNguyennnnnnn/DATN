#!/usr/bin/env python3
"""So sánh ArcFace / FLAME: cùng người khác biểu cảm vs khác người cùng biểu cảm.

Đọc hybrid context từ LMDB (mặc định slat_context_balanced.lmdb).
In vài cặp mẫu + thống kê cos cho FaceVerse và FaceScape.
"""
from __future__ import annotations

import argparse
import io
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.imf_diffusion import slice_contrastive_context

ARC_DIM, FLAME_DIM = 512, 50


@dataclass
class Sample:
    key: str
    subject: str
    expression: str
    arc: torch.Tensor
    flame: torch.Tensor


def load_ctx_blob(raw: bytes) -> torch.Tensor:
    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        return obj["context"].float().flatten()
    return obj.float().flatten()


def parse_faceverse(key: str) -> tuple[str, str]:
    # faceverse/011_01/011_01.obj
    rel = key.split("/", 1)[1]
    folder = rel.split("/")[0]
    m = re.match(r"^(\d+)_(\d+)$", folder)
    if m:
        return m.group(1), m.group(2)
    # fallback: 011_01 -> 011, 01
    parts = folder.split("_", 1)
    return parts[0], parts[1] if len(parts) > 1 else "0"


def parse_facescape(key: str) -> tuple[str, str]:
    # facescape/100/models_reg/10_dimpler.obj
    rel = key.split("/", 1)[1]
    parts = rel.split("/")
    subject = parts[0]
    expr = os.path.splitext(parts[-1])[0]
    return subject, expr


def cos_pair(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float().flatten(), dim=0)
    b = F.normalize(b.float().flatten(), dim=0)
    return float((a @ b).item())


def load_dataset_samples(
    env: lmdb.Environment,
    prefix: str,
    parser: Callable[[str], tuple[str, str]],
    max_per_subject: int,
    seed: int,
) -> list[Sample]:
    rng = random.Random(seed)
    by_subj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with env.begin() as txn:
        for k, _ in txn.cursor():
            if k == b"__meta__":
                continue
            key = k.decode()
            if not key.startswith(prefix):
                continue
            subj, expr = parser(key)
            by_subj[subj].append((key, expr))

    samples: list[Sample] = []
    with env.begin() as txn:
        for subj, items in by_subj.items():
            # dedupe expressions
            seen_expr: dict[str, str] = {}
            for key, expr in items:
                if expr not in seen_expr:
                    seen_expr[expr] = key
            keys = list(seen_expr.values())
            if len(keys) > max_per_subject:
                keys = rng.sample(keys, max_per_subject)
            for key in keys:
                raw = txn.get(key.encode())
                if raw is None:
                    continue
                ctx = load_ctx_blob(raw)
                arc = slice_contrastive_context(ctx.unsqueeze(0), "arcface").squeeze(0)
                flame = slice_contrastive_context(ctx.unsqueeze(0), "flame").squeeze(0)
                _, expr = parser(key)
                samples.append(Sample(key=key, subject=subj, expression=expr, arc=arc, flame=flame))
    return samples


def pick_same_person_pairs(
    samples: list[Sample],
    n_pairs: int,
    rng: random.Random,
) -> list[tuple[Sample, Sample, float, float]]:
    by_subj: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_subj[s.subject].append(s)
    eligible = [v for v in by_subj.values() if len(v) >= 2]
    if not eligible:
        return []
    pairs: list[tuple[Sample, Sample, float, float]] = []
    tries = 0
    while len(pairs) < n_pairs and tries < n_pairs * 50:
        tries += 1
        group = rng.choice(eligible)
        a, b = rng.sample(group, 2)
        if a.expression == b.expression:
            continue
        pairs.append(
            (
                a,
                b,
                cos_pair(a.arc, b.arc),
                cos_pair(a.flame, b.flame),
            )
        )
    return pairs


def pick_same_expr_diff_person(
    samples: list[Sample],
    n_pairs: int,
    rng: random.Random,
) -> list[tuple[Sample, Sample, float, float]]:
    by_expr: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_expr[s.expression].append(s)
    eligible = [v for v in by_expr.values() if len(v) >= 2]
    pairs: list[tuple[Sample, Sample, float, float]] = []
    tries = 0
    while len(pairs) < n_pairs and tries < n_pairs * 80:
        tries += 1
        if not eligible:
            break
        group = rng.choice(eligible)
        a, b = rng.sample(group, 2)
        if a.subject == b.subject:
            continue
        pairs.append(
            (
                a,
                b,
                cos_pair(a.arc, b.arc),
                cos_pair(a.flame, b.flame),
            )
        )
    return pairs


def pick_diff_person_diff_expr(
    samples: list[Sample],
    n_pairs: int,
    rng: random.Random,
) -> list[tuple[Sample, Sample, float, float]]:
    pairs: list[tuple[Sample, Sample, float, float]] = []
    if len(samples) < 2:
        return pairs
    tries = 0
    while len(pairs) < n_pairs and tries < n_pairs * 30:
        tries += 1
        a, b = rng.sample(samples, 2)
        if a.subject == b.subject:
            continue
        pairs.append(
            (
                a,
                b,
                cos_pair(a.arc, b.arc),
                cos_pair(a.flame, b.flame),
            )
        )
    return pairs


def summarize(name: str, arc_vals: list[float], flame_vals: list[float]) -> None:
    def stat(xs: list[float]) -> str:
        if not xs:
            return "n=0"
        a = np.array(xs)
        return f"n={len(xs)} mean={a.mean():.4f} std={a.std():.4f} min={a.min():.4f} max={a.max():.4f}"

    print(f"\n  [{name}]")
    print(f"    ArcFace cos:  {stat(arc_vals)}")
    print(f"    FLAME cos:    {stat(flame_vals)}")


def print_pairs(title: str, pairs: list[tuple[Sample, Sample, float, float]], max_show: int) -> None:
    print(f"\n{'─'*72}")
    print(f"  {title} (hiển thị {min(max_show, len(pairs))}/{len(pairs)} cặp)")
    print(f"{'─'*72}")
    for i, (a, b, arc_c, fl_c) in enumerate(pairs[:max_show]):
        print(f"\n  Cặp {i + 1}: ArcFace cos={arc_c:.4f}  FLAME cos={fl_c:.4f}")
        print(f"    A: subj={a.subject} expr={a.expression}")
        print(f"       {a.key}")
        print(f"    B: subj={b.subject} expr={b.expression}")
        print(f"       {b.key}")


def analyze_dataset(
    name: str,
    samples: list[Sample],
    n_pairs: int,
    n_show: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    n_subj = len({s.subject for s in samples})
    n_expr = len({s.expression for s in samples})
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"  samples={len(samples)}  subjects={n_subj}  unique_expr={n_expr}")
    print(f"{'='*72}")

    same_person = pick_same_person_pairs(samples, n_pairs, rng)
    same_expr = pick_same_expr_diff_person(samples, n_pairs, rng)
    diff_both = pick_diff_person_diff_expr(samples, n_pairs, rng)

    print_pairs(
        "A) CÙNG người, KHÁC biểu cảm — ArcFace nên CAO, FLAME thấp hơn same-expr",
        same_person,
        n_show,
    )
    print_pairs(
        "B) KHÁC người, CÙNG biểu cảm — FLAME nên CAO hơn A; ArcFace nên THẤP hơn A",
        same_expr,
        n_show,
    )
    print_pairs(
        "C) KHÁC người, KHÁC biểu cảm (baseline)",
        diff_both,
        n_show,
    )

    summarize(
        "A) same person, diff expr",
        [p[2] for p in same_person],
        [p[3] for p in same_person],
    )
    summarize(
        "B) diff person, same expr",
        [p[2] for p in same_expr],
        [p[3] for p in same_expr],
    )
    summarize(
        "C) diff person, diff expr",
        [p[2] for p in diff_both],
        [p[3] for p in diff_both],
    )

    if same_person and same_expr:
        arc_a = np.mean([p[2] for p in same_person])
        arc_b = np.mean([p[2] for p in same_expr])
        fl_a = np.mean([p[3] for p in same_person])
        fl_b = np.mean([p[3] for p in same_expr])
        print(f"\n  [Tóm tắt {name}]")
        print(f"    ArcFace: same-person={arc_a:.4f}  same-expr-diff-person={arc_b:.4f}  Δ={arc_a - arc_b:+.4f}")
        print(f"    FLAME:   same-person={fl_a:.4f}  same-expr-diff-person={fl_b:.4f}  Δ={fl_b - fl_a:+.4f}")
        print(
            "    Kỳ vọng: Arc(same person) > Arc(same expr cross-person); "
            "FLAME(same expr) > FLAME(same person) nếu FLAME encode pose tốt."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="So sánh ArcFace/FLAME theo cặp identity vs expression")
    ap.add_argument("--lmdb", default="data/slat_context_balanced.lmdb")
    ap.add_argument("--pairs", type=int, default=30, help="Số cặp mỗi loại mỗi dataset")
    ap.add_argument("--show", type=int, default=5, help="Số cặp in chi tiết")
    ap.add_argument("--max-per-subject", type=int, default=25, help="Giới hạn mẫu/subject khi load")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.isdir(args.lmdb):
        raise SystemExit(f"LMDB not found: {args.lmdb}")

    print("=" * 72)
    print("  SO SÁNH CONTEXT: ArcFace vs FLAME")
    print(f"  LMDB: {args.lmdb}")
    print("=" * 72)

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False)

    fv_samples = load_dataset_samples(
        env, "faceverse/", parse_faceverse, args.max_per_subject, args.seed,
    )
    fs_samples = load_dataset_samples(
        env, "facescape/", parse_facescape, args.max_per_subject, args.seed + 1,
    )
    env.close()

    analyze_dataset("FaceVerse", fv_samples, args.pairs, args.show, args.seed + 10)
    analyze_dataset("FaceScape", fs_samples, args.pairs, args.show, args.seed + 20)

    print("\n" + "=" * 72)
    print("  GHI CHÚ")
    print("  - LMDB balanced: mỗi khối Arc/FLAME/DINO đã L2 norm (~||·||≈1 mỗi segment).")
    print("  - ArcFace-only vào v8 lite: chỉ [:512] vào model; FLAME chỉ để phân tích ở đây.")
    print("=" * 72)


if __name__ == "__main__":
    main()
