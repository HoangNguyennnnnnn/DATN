"""
FLAME Expression Adapter — MediaPipe FaceLandmarker (v2 with blendshapes).

Output: 50-dim expression vector from rendered face image.

Source: MediaPipe v2 outputs 52 ARKit-compatible blendshape scores per face.
Chúng ta drop 2 blendshapes (`_neutral` luôn = 0, và `mouthClose` thường
redundant với jaw) để giữ context_dim=946 = ArcFace(512) + FLAME(50) + DINOv2(384).

Model file: `data/mediapipe_models/face_landmarker_v2_with_blendshapes.task`
Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Differences vs cũ:
- KHÔNG còn random-init CNN (cũ output gần như constant → bug)
- Output thực tế VARYING theo expression (verified: cos_sim 0.39-0.72 cross-expr)
- CPU/GPU-agnostic (MediaPipe tự dispatch)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


_MEDIAPIPE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "mediapipe_models",
    "face_landmarker_v2_with_blendshapes.task",
)
_MEDIAPIPE_MODEL_PATH = os.path.abspath(_MEDIAPIPE_MODEL_PATH)

# Drop 2 indices từ 52 blendshapes của MediaPipe để giữ 50-dim:
#   index 0: "_neutral" — luôn = 0, không thông tin
#   index 1: "browDownLeft" — chọn drop để pair với browDownRight (52 → 50)
#     (Alternative: take all 52, change context_dim → 948. Nhưng giữ 946 đơn giản hơn.)
_DROP_INDICES = (0, 1)
_KEEP_MASK = np.ones(52, dtype=bool)
_KEEP_MASK[list(_DROP_INDICES)] = False


class FLAMEExpressionAdapter(nn.Module):
    """Trích xuất expression vector từ image dùng MediaPipe FaceLandmarker V2 (52 blendshapes).

    Giữ interface giống cũ (`extract_from_image`, `extract_from_vertices`) để
    `build_context_lmdb.py` và `train_imf.py` không cần đổi.

    `extract_from_vertices` không support trong implementation mới —
    sẽ raise NotImplementedError. Nếu cần, dùng vertex → render image → extract.
    """

    def __init__(self, expression_dim: int = 50, device: str = "cuda:0",
                 model_path: Optional[str] = None):
        super().__init__()
        if expression_dim != 50:
            raise ValueError(
                f"FLAMEExpressionAdapter (MediaPipe variant) yêu cầu expression_dim=50 "
                f"(52 blendshapes − 2 dropped). Got {expression_dim}."
            )
        self.expression_dim = expression_dim
        # PyTorch device parameter giữ để API tương thích, nhưng MediaPipe tự dispatch
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")

        # Lazy import — không ép user phải có mediapipe khi không cần
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise ImportError(
                "MediaPipe required for FLAMEExpressionAdapter. "
                "Install: pip install mediapipe"
            ) from e

        path = model_path or _MEDIAPIPE_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"MediaPipe model not found at {path}. Download:\n"
                f"  mkdir -p data/mediapipe_models\n"
                f"  wget -O {path} "
                f"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

        param_count = sum(p.numel() for p in self.parameters())
        print(f"[FLAME] MediaPipe FaceLandmarker V2 initialized: {expression_dim}-dim "
              f"(52 blendshapes - 2 dropped)")
        print(f"[FLAME] Model: {os.path.basename(path)}, "
              f"trainable params: {param_count} (ext model frozen)")

    @torch.no_grad()
    def extract_from_image(self, face_image: torch.Tensor) -> torch.Tensor:
        """Extract 50-dim expression từ ảnh khuôn mặt.

        Args:
            face_image: tensor shape [1, 3, H, W] hoặc [3, H, W] hoặc [H, W, 3].
                Giá trị có thể là [0, 1] float hoặc [0, 255] uint8.

        Returns:
            [1, 50] expression vector.
            Nếu không detect được face: trả zero vector + warn (không fallback ngẫu nhiên).
        """
        arr = face_image.detach().cpu().numpy() if hasattr(face_image, "detach") else np.asarray(face_image)

        # Squeeze batch dim
        while arr.ndim == 4:
            arr = arr[0]
        # Transpose CHW → HWC
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = arr.transpose(1, 2, 0)
        # RGBA → RGB
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        # Scale to uint8
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=arr)
        result = self._detector.detect(mp_image)

        if not result.face_blendshapes:
            # No face detected — trả zero (caller có thể fallback hoặc bỏ qua sample)
            return torch.zeros(1, self.expression_dim, dtype=torch.float32, device=self.device)

        scores = np.array([b.score for b in result.face_blendshapes[0]], dtype=np.float32)
        # Drop 2 redundant indices (0=_neutral, 1=browDownLeft) → 50-dim
        scores_50 = scores[_KEEP_MASK]
        assert scores_50.shape == (self.expression_dim,), \
            f"Expected {self.expression_dim} blendshapes, got {scores_50.shape}"

        return torch.from_numpy(scores_50).to(self.device).unsqueeze(0)

    @torch.no_grad()
    def extract_from_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        """Không support trong MediaPipe variant.

        MediaPipe yêu cầu rendered face image. Nếu chỉ có vertices,
        cần render trước (e.g., via `src/data/mesh_renderer.py`) rồi gọi
        `extract_from_image`.
        """
        raise NotImplementedError(
            "FLAMEExpressionAdapter (MediaPipe variant) chỉ support extract_from_image. "
            "Để dùng vertices, hãy render mesh trước qua MeshRenderer.render_front_and_back()."
        )

    def forward(self, face_image: torch.Tensor) -> torch.Tensor:
        return self.extract_from_image(face_image)

    def __del__(self):
        # MediaPipe detector cần explicit close để tránh leak resources
        try:
            if hasattr(self, "_detector") and self._detector is not None:
                self._detector.close()
        except Exception:
            pass


def balance_hybrid_context_segments(
    identity: torch.Tensor,
    expression: torch.Tensor,
    back_shape: Optional[torch.Tensor] = None,
    *,
    arc_dim: int = 512,
    flame_dim: int = 50,
) -> torch.Tensor:
    """L2-normalize từng khối trước khi concat — tránh DINO (~‖·‖≈46) lấn át ArcFace (~1).

    2026-05-22: Audit phát hiện ArcFace chỉ ~0.05% năng lượng của vector 946-d thô
    → context_cond_mlp gần như chỉ thấy DINO → model bỏ qua identity.
    """
    import torch.nn.functional as F

    id_n = identity.reshape(-1, arc_dim)
    id_n = F.normalize(id_n, p=2, dim=-1)

    ex = expression.reshape(-1, flame_dim)
    ex_norm = ex.norm(dim=-1, keepdim=True)
    ex_n = torch.where(ex_norm > 1e-6, ex / ex_norm.clamp(min=1e-8), ex)

    if back_shape is None:
        return torch.cat([id_n, ex_n], dim=-1)

    d_n = back_shape.reshape(-1, back_shape.shape[-1])
    d_n = F.normalize(d_n, p=2, dim=-1)
    return torch.cat([id_n, ex_n, d_n], dim=-1)


def create_hybrid_context(
    identity: torch.Tensor,
    expression: torch.Tensor,
    back_shape: Optional[torch.Tensor] = None,
    balance_segments: bool = True,
) -> torch.Tensor:
    """Kết hợp ArcFace [512] + FLAME [50] + DINOv2_Back [384] → [946]."""
    if balance_segments:
        return balance_hybrid_context_segments(identity, expression, back_shape)
    if back_shape is None:
        return torch.cat([identity, expression], dim=-1)
    return torch.cat([identity, expression, back_shape], dim=-1)


if __name__ == "__main__":
    # Smoke test
    print("Smoke testing FLAMEExpressionAdapter (MediaPipe variant)...")
    flame = FLAMEExpressionAdapter(expression_dim=50, device="cpu")
    fake_img = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    out = flame.extract_from_image(fake_img)
    print(f"  output shape: {out.shape}  (expected [1, 50])")
    print(f"  no face → zero vector: {torch.all(out == 0).item()}")
    print("  ✓ Smoke test passed")
