"""
CLI: Image → 946-dim context tensor (.pt file).

Examples:
  # Chỉ ảnh frontal (DINOv2 sẽ là zero vector):
  python scripts/preprocess_image.py --input photo.jpg --output ctx.pt

  # Có cả back view:
  python scripts/preprocess_image.py --input photo.jpg --back back.jpg --output ctx.pt

  # Save debug images (crop, mask):
  python scripts/preprocess_image.py --input photo.jpg --output ctx.pt --debug-dir /tmp/dbg/
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.image_preprocessor import ImagePreprocessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Ảnh frontal (jpg/png).")
    ap.add_argument("--back", default=None, help="Ảnh back-of-head (optional).")
    ap.add_argument("--output", required=True, help="Output .pt path.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--debug-dir", default=None, help="Lưu ảnh debug (crop, mask).")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image không tồn tại: {args.input}")
    if args.back is not None and not os.path.exists(args.back):
        raise FileNotFoundError(f"Back image không tồn tại: {args.back}")

    print(f"[1/3] Loading preprocessor (device={args.device})...")
    t0 = time.time()
    pp = ImagePreprocessor(device=args.device)
    print(f"       loaded in {time.time() - t0:.1f}s")

    print(f"[2/3] Processing image: {args.input}")
    if args.back:
        print(f"       back: {args.back}")
    t1 = time.time()
    ctx = pp.process(
        image_path=args.input,
        back_image_path=args.back,
        save_debug_dir=args.debug_dir,
    )
    dt = time.time() - t1
    print(f"       done in {dt * 1000:.0f} ms")

    print(f"[3/3] Saving: {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    torch.save(
        {
            "context": ctx.cpu(),
            "source": os.path.abspath(args.input),
            "back": os.path.abspath(args.back) if args.back else None,
        },
        args.output,
    )

    print(f"\n=== Summary ===")
    print(f"  context shape : {tuple(ctx.shape)}")
    print(f"  arcface norm  : {ctx[:512].norm().item():.4f}")
    print(f"  flame   sum   : {ctx[512:562].sum().item():.4f}")
    print(f"  dino    norm  : {ctx[562:].norm().item():.4f}")
    print(f"  output        : {args.output}")
    if args.debug_dir:
        print(f"  debug dir     : {args.debug_dir}")


if __name__ == "__main__":
    main()
