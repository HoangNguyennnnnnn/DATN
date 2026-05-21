"""
Image → 946-dim Context Preprocessor.

Pipeline cho ứng dụng thực tế: nhận ảnh raw (background bất kỳ), tự cắt nền
giữ face + vai, extract 3 thành phần context, trả tensor [946] sẵn sàng
feed VoxelMamba+iMF.

Context layout:
    [   0 :  512] ArcFace identity (L2-normalized)
    [ 512 :  562] MediaPipe FaceLandmarker blendshapes (50)
    [ 562 :  946] DINOv2 back-of-head features (384) — 0 nếu không có ảnh back
"""
from __future__ import annotations

import os
import sys
import urllib.request
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Allow running both as module và direct
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, "..", "..")))

from src.data.arcface_extractor import ArcFaceExtractor
from src.data.flame_adapter import FLAMEExpressionAdapter
from src.data.feature_extractor import DinoV3Extractor


# ============================================================
# MediaPipe Selfie Multiclass Segmenter — auto-download
# ============================================================
_SEG_MODEL_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "data", "mediapipe_models"))
_SEG_MODEL_PATH = os.path.join(_SEG_MODEL_DIR, "selfie_multiclass_256x256.tflite")
_SEG_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)

# Selfie Multiclass output: 6 lớp
#   0: background, 1: hair, 2: body-skin, 3: face-skin, 4: clothes, 5: others
_BG_CLASS = 0


def _ensure_segmenter_model() -> str:
    if os.path.exists(_SEG_MODEL_PATH):
        return _SEG_MODEL_PATH
    os.makedirs(_SEG_MODEL_DIR, exist_ok=True)
    print(f"[ImagePreprocessor] Downloading segmenter model → {_SEG_MODEL_PATH} (~250 KB)")
    urllib.request.urlretrieve(_SEG_MODEL_URL, _SEG_MODEL_PATH)
    return _SEG_MODEL_PATH


# ============================================================
# Helpers
# ============================================================
def _load_image_rgb(path: str) -> np.ndarray:
    """Đọc ảnh → numpy RGB uint8 [H, W, 3]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _resize_min_side(img: np.ndarray, min_side: int) -> np.ndarray:
    """Resize giữ aspect ratio sao cho cạnh ngắn ≥ min_side."""
    h, w = img.shape[:2]
    short = min(h, w)
    if short >= min_side:
        return img
    scale = min_side / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return np.asarray(
        Image.fromarray(img).resize((new_w, new_h), Image.BICUBIC),
        dtype=np.uint8,
    )


def _shoulders_bbox(
    face_bbox: Tuple[float, float, float, float],
    img_h: int,
    img_w: int,
    up: float = 0.8,
    down: float = 2.5,
    side: float = 1.0,
) -> Tuple[int, int, int, int]:
    """Mở rộng face bbox để bao luôn vai + đỉnh đầu. Clamp về bounds."""
    x1, y1, x2, y2 = face_bbox
    fw, fh = x2 - x1, y2 - y1
    nx1 = int(round(x1 - side * fw))
    nx2 = int(round(x2 + side * fw))
    ny1 = int(round(y1 - up * fh))
    ny2 = int(round(y2 + down * fh))
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)
    return nx1, ny1, nx2, ny2


def _np_to_tensor_chw(img_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """[H, W, 3] uint8 → [1, 3, H, W] float [0, 1] on device."""
    t = torch.from_numpy(img_uint8).to(device, dtype=torch.float32) / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).contiguous()


def _save_debug(img: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(path)


# ============================================================
# Main class
# ============================================================
class ImagePreprocessor:
    """Image → 946-dim context tensor.

    Reuses 3 extractors có sẵn (ArcFace, FLAME, DINOv2) + MediaPipe ImageSegmenter
    (Selfie Multiclass). VRAM ~1.5 GB, latency ~200-250 ms/ảnh (GPU).
    """

    def __init__(self, device: str = "cuda:0", bg_color: Tuple[int, int, int] = (128, 128, 128),
                 dino_back_mean_path: Optional[str] = None):
        self.device = torch.device(device)
        self.bg_color = np.array(bg_color, dtype=np.uint8)

        print(f"[ImagePreprocessor] Initializing on {self.device}...")
        self.arcface = ArcFaceExtractor(device=str(device))
        self.flame = FLAMEExpressionAdapter(device=str(device))
        self.dino = DinoV3Extractor(device=str(device))
        self._init_segmenter()

        # Load DINOv2 back fallback (mean từ training distribution).
        # Tránh distribution shift: training data luôn có DINOv2 ≠ 0 (norm~46),
        # nên zero fallback đẩy context OOD → model collapse.
        if dino_back_mean_path is None:
            dino_back_mean_path = os.path.abspath(
                os.path.join(_THIS_DIR, "..", "..", "data", "dino_back_mean.pt")
            )
        if os.path.exists(dino_back_mean_path):
            _blob = torch.load(dino_back_mean_path, map_location="cpu", weights_only=False)
            self.dino_back_mean = _blob["mean"].to(self.device, dtype=torch.float32)
            print(f"[ImagePreprocessor] Loaded DINOv2 back fallback "
                  f"(norm={self.dino_back_mean.norm().item():.2f}) from {dino_back_mean_path}")
        else:
            self.dino_back_mean = None
            warnings.warn(
                f"[ImagePreprocessor] No DINOv2 back mean at {dino_back_mean_path}. "
                "Falling back to zeros — risk of distribution shift if back image not provided."
            )

        print("[ImagePreprocessor] Ready.")

    def _init_segmenter(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise ImportError(
                "MediaPipe required. Install: pip install mediapipe"
            ) from e

        model_path = _ensure_segmenter_model()
        options = mp_vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_category_mask=True,
            output_confidence_masks=False,
        )
        self._segmenter = mp_vision.ImageSegmenter.create_from_options(options)
        self._mp = mp

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------
    def _segment_remove_bg(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Selfie Multiclass segmentation. Trả (composed_img, fg_mask_bool).

        Composed image = giữ foreground, thay background bằng `self.bg_color`.
        """
        try:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(img_rgb),
            )
            result = self._segmenter.segment(mp_image)
            cat_mask = np.asarray(result.category_mask.numpy_view())
            if cat_mask.ndim > 2:
                cat_mask = cat_mask.squeeze()  # [H, W] uint8 class indices
        except Exception as e:
            warnings.warn(f"[ImagePreprocessor] Segmentation failed ({e}); keep original bg.")
            return img_rgb, np.ones(img_rgb.shape[:2], dtype=bool)

        fg = cat_mask != _BG_CLASS  # [H, W] bool
        if fg.sum() < 100:  # ít foreground quá → segmenter lỗi
            warnings.warn("[ImagePreprocessor] Very small foreground mask; keep original bg.")
            return img_rgb, np.ones(img_rgb.shape[:2], dtype=bool)

        composed = img_rgb.copy()
        bg_pixels = ~fg
        composed[bg_pixels] = self.bg_color
        return composed, fg

    def _bbox_from_mask(self, mask: np.ndarray, pad: int = 8) -> Tuple[int, int, int, int]:
        """Tight bbox của mask=True. Pad rồi clamp."""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            h, w = mask.shape
            return 0, 0, w, h
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        h, w = mask.shape
        return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad + 1), min(h, y2 + pad + 1)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    @torch.no_grad()
    def process(
        self,
        image_path: str,
        back_image_path: Optional[str] = None,
        save_debug_dir: Optional[str] = None,
    ) -> torch.Tensor:
        """Trả về context tensor `[946]` trên `self.device`."""
        # ---- 1. Frontal: load + ensure min size ----
        front = _load_image_rgb(image_path)
        front = _resize_min_side(front, min_side=512)

        # ---- 2. Face detection ----
        det = self.arcface.detect_face(front)
        if det is None:
            raise ValueError(
                f"Không phát hiện được khuôn mặt rõ trong ảnh frontal: {image_path}"
            )

        # ---- 3. Crop face + shoulders ----
        x1, y1, x2, y2 = _shoulders_bbox(det["bbox"], front.shape[0], front.shape[1])
        front_crop = front[y1:y2, x1:x2].copy()
        if front_crop.size == 0:
            raise ValueError("Shoulders bbox rỗng sau khi clamp.")

        # ---- 4. Background removal (segmentation) ----
        front_clean, _ = self._segment_remove_bg(front_crop)

        # ---- 5. ArcFace embedding ----
        # detect_face đã trả embedding cho ảnh full, dùng luôn — tránh chạy 2 lần.
        arcface_emb = det["embedding"].to(self.device)  # [512]

        # ---- 6. FLAME blendshapes (cần ảnh đã clean để khớp với training context) ----
        flame_in = _np_to_tensor_chw(front_clean, self.device)
        flame_emb = self.flame.extract_from_image(flame_in).squeeze(0).to(self.device)  # [50]

        # ---- 7. DINOv2 from back image (optional) ----
        if back_image_path is not None and os.path.exists(back_image_path):
            back = _load_image_rgb(back_image_path)
            back = _resize_min_side(back, min_side=224)
            back_clean, fg_mask = self._segment_remove_bg(back)
            bx1, by1, bx2, by2 = self._bbox_from_mask(fg_mask, pad=16)
            back_crop = back_clean[by1:by2, bx1:bx2]
            if back_crop.size == 0:
                back_crop = back_clean
            back_in = _np_to_tensor_chw(back_crop, self.device)
            dino_emb = self.dino.extract_features(back_in).squeeze(0).to(self.device).float()  # [384]
        else:
            # Fallback: dùng training mean DINOv2 (norm~46) thay vì zeros để tránh
            # distribution shift. Training data ALWAYS có DINOv2 ≠ 0; zero fallback
            # đẩy context OOD → model collapse (sinh ra cube).
            if self.dino_back_mean is not None:
                dino_emb = self.dino_back_mean.clone().to(self.device).float()
            else:
                dino_emb = torch.zeros(384, dtype=torch.float32, device=self.device)
            back_clean = None

        # ---- 8. Concat → [946] ----
        context = torch.cat(
            [arcface_emb.float(), flame_emb.float(), dino_emb.float()], dim=0
        )
        assert context.shape == (946,), f"Expected [946], got {context.shape}"

        # ---- 9. Debug dump ----
        if save_debug_dir is not None:
            os.makedirs(save_debug_dir, exist_ok=True)
            _save_debug(front_crop, os.path.join(save_debug_dir, "01_front_crop.png"))
            _save_debug(front_clean, os.path.join(save_debug_dir, "02_front_clean.png"))
            if back_clean is not None:
                _save_debug(back_clean, os.path.join(save_debug_dir, "03_back_clean.png"))
            with open(os.path.join(save_debug_dir, "bbox.txt"), "w") as f:
                f.write(f"face_bbox={det['bbox']}\nshoulders_bbox=[{x1},{y1},{x2},{y2}]\n")

        return context


# ============================================================
# Standalone smoke test
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--back", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--debug-dir", default=None)
    args = parser.parse_args()

    p = ImagePreprocessor(device=args.device)
    ctx = p.process(args.input, back_image_path=args.back, save_debug_dir=args.debug_dir)
    print(f"context shape: {tuple(ctx.shape)}")
    print(f"  arcface norm: {ctx[:512].norm():.4f}")
    print(f"  flame   sum : {ctx[512:562].sum():.4f}")
    print(f"  dino    norm: {ctx[562:].norm():.4f}")
