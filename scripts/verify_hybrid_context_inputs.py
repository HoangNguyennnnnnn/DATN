#!/usr/bin/env python3
"""
Render and verify front/back images before ArcFace / FLAME / DINOv2.

This script:
1. Renders front/back images from a mesh.
2. Saves the raw render inputs so they can be inspected visually.
3. Runs ArcFace, FLAME, and DINOv2 once to verify the actual module path.
4. Writes a small Markdown report with shapes, providers, and output artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from src.data.arcface_extractor import ArcFaceExtractor
from src.data.feature_extractor import DinoV3Extractor
from src.data.flame_adapter import FLAMEExpressionAdapter
from src.data.mesh_renderer import MeshRenderer


def _tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    x = tensor.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    x = x.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)


def _save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(_tensor_to_uint8_image(tensor)).save(path)


def _save_contact_sheet(front: torch.Tensor, back: torch.Tensor, path: str) -> None:
    front_img = Image.fromarray(_tensor_to_uint8_image(front))
    back_img = Image.fromarray(_tensor_to_uint8_image(back))
    w, h = front_img.size
    canvas = Image.new("RGB", (w * 2, h + 36), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(front_img, (0, 36))
    canvas.paste(back_img, (w, 36))
    draw.text((16, 10), "Front -> ArcFace / FLAME", fill=(0, 0, 0))
    draw.text((w + 16, 10), "Back -> DINOv2", fill=(0, 0, 0))
    canvas.save(path)


def _draw_arcface_bbox(front: torch.Tensor, extractor: ArcFaceExtractor, path: str) -> dict[str, Any]:
    image_rgb = _tensor_to_uint8_image(front)
    out = Image.fromarray(image_rgb)
    info: dict[str, Any] = {"face_count": 0, "largest_bbox": None}
    if getattr(extractor, "mode", None) != "insightface":
        out.save(path)
        return info

    image_bgr = image_rgb[:, :, ::-1].copy()
    faces = extractor.app.get(image_bgr)
    draw = ImageDraw.Draw(out)
    info["face_count"] = len(faces)
    if faces:
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        bbox = [float(v) for v in largest.bbox.tolist()]
        info["largest_bbox"] = bbox
        draw.rectangle(bbox, outline=(255, 0, 0), width=3)
    out.save(path)
    return info


def _save_flame_input_preview(front: torch.Tensor, path: str) -> tuple[int, int]:
    resized = F.interpolate(front.float(), size=(112, 112), mode="bilinear", align_corners=False)
    _save_tensor_image(resized, path)
    return (112, 112)


def _save_dino_input_preview(back: torch.Tensor, extractor: DinoV3Extractor, path: str) -> dict[str, Any]:
    inputs = extractor.processor(images=back, return_tensors="pt", do_rescale=False)
    pixel_values = inputs["pixel_values"][0].detach().cpu().float()
    mean = torch.tensor(getattr(extractor.processor, "image_mean", [0.485, 0.456, 0.406]), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(getattr(extractor.processor, "image_std", [0.229, 0.224, 0.225]), dtype=torch.float32).view(3, 1, 1)
    preview = (pixel_values * std + mean).clamp(0.0, 1.0).unsqueeze(0)
    _save_tensor_image(preview, path)
    return {
        "processor_backend": getattr(extractor, "processor_backend", "unknown"),
        "processor_size": getattr(extractor.processor, "size", None),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
    }


def _report_lines(data: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Context Render Verification",
        "",
        f"- Timestamp: {data['timestamp']}",
        f"- Mesh: `{data['obj_path']}`",
        f"- Device: `{data['device']}`",
        f"- Renderer image size: `{data['renderer_image_size']}`",
        "",
        "## Raw Render Inputs",
        f"- Front raw image: `{data['front_image_path']}`",
        f"- Back raw image: `{data['back_image_path']}`",
        f"- Contact sheet: `{data['contact_sheet_path']}`",
        f"- Front shape: `{data['front_shape']}`",
        f"- Back shape: `{data['back_shape']}`",
        "",
        "## ArcFace / FLAME / DINOv2",
        f"- ArcFace mode: `{data['arcface_mode']}`",
        f"- ArcFace providers: `{data['arcface_providers']}`",
        f"- ArcFace face count on front render: `{data['arcface_face_count']}`",
        f"- ArcFace bbox preview: `{data['arcface_bbox_path']}`",
        f"- ArcFace embedding shape: `{data['arcface_embedding_shape']}`",
        f"- ArcFace embedding norm: `{data['arcface_embedding_norm']:.6f}`",
        f"- FLAME input preview: `{data['flame_input_path']}`",
        f"- FLAME embedding shape: `{data['flame_shape']}`",
        f"- DINO processor backend: `{data['dino_processor_backend']}`",
        f"- DINO input preview: `{data['dino_input_path']}`",
        f"- DINO embedding shape: `{data['dino_shape']}`",
        "",
        "## Fallback Status",
        f"- ArcFace code fallback active: `{data['arcface_mode'] != 'insightface'}`",
        f"- DINO model fallback active: `False`",
        f"- FLAME fallback active: `False`",
        "",
        "## Notes",
        "- ArcFace and FLAME consume the front render.",
        "- DINOv2 consumes the back render.",
        "- The saved raw renders are the images to inspect for framing, rotation, and scale before they enter the downstream models.",
    ]
    if data.get("notes"):
        lines.append(f"- Runtime note: {data['notes']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify front/back render inputs for ArcFace/FLAME/DINOv2")
    parser.add_argument("--obj-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/verification_hybrid_context")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    stem = os.path.splitext(os.path.basename(args.obj_path))[0]
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"{run_ts}_{stem}")
    os.makedirs(out_dir, exist_ok=True)

    renderer = MeshRenderer(device=device, image_size=int(args.image_size))
    front, back = renderer.render_front_and_back(args.obj_path)

    front_path = os.path.join(out_dir, "front_raw.png")
    back_path = os.path.join(out_dir, "back_raw.png")
    contact_path = os.path.join(out_dir, "front_back_contact_sheet.png")
    _save_tensor_image(front, front_path)
    _save_tensor_image(back, back_path)
    _save_contact_sheet(front, back, contact_path)

    arcface = ArcFaceExtractor(device=device)
    flame = FLAMEExpressionAdapter(expression_dim=50, device=device)
    dino = DinoV3Extractor(model_name="facebook/dinov2-small", device=device)

    bbox_path = os.path.join(out_dir, "front_arcface_bbox.png")
    bbox_info = _draw_arcface_bbox(front, arcface, bbox_path)

    arcface_embedding = arcface.extract_identity(front)
    flame_embedding = flame.extract_from_image(front)
    dino_embedding = dino.extract_features(back)

    flame_input_path = os.path.join(out_dir, "front_flame_input_112.png")
    _save_flame_input_preview(front, flame_input_path)

    dino_input_path = os.path.join(out_dir, "back_dino_processed.png")
    dino_input_info = _save_dino_input_preview(back, dino, dino_input_path)

    report_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "obj_path": args.obj_path,
        "device": device,
        "renderer_image_size": int(args.image_size),
        "front_image_path": front_path,
        "back_image_path": back_path,
        "contact_sheet_path": contact_path,
        "front_shape": list(front.shape),
        "back_shape": list(back.shape),
        "arcface_mode": getattr(arcface, "mode", "unknown"),
        "arcface_providers": getattr(arcface, "providers", []),
        "arcface_face_count": int(bbox_info["face_count"]),
        "arcface_bbox_path": bbox_path,
        "arcface_embedding_shape": list(arcface_embedding.shape),
        "arcface_embedding_norm": float(arcface_embedding.norm(dim=-1).mean().item()),
        "flame_input_path": flame_input_path,
        "flame_shape": list(flame_embedding.shape),
        "dino_processor_backend": dino_input_info["processor_backend"],
        "dino_input_path": dino_input_path,
        "dino_shape": list(dino_embedding.shape),
        "notes": None,
    }

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_report_lines(report_data))

    summary_path = os.path.join(out_dir, "report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"[VerifyHybridContext] front={front_path}")
    print(f"[VerifyHybridContext] back={back_path}")
    print(f"[VerifyHybridContext] contact={contact_path}")
    print(f"[VerifyHybridContext] report={report_path}")
    print(f"[VerifyHybridContext] arcface_mode={report_data['arcface_mode']} providers={report_data['arcface_providers']}")
    print(f"[VerifyHybridContext] flame_shape={report_data['flame_shape']} dino_shape={report_data['dino_shape']}")


if __name__ == "__main__":
    main()
