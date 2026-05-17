"""Test MediaPipe FLAME extractor trên các expressions khác nhau."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.data.mesh_renderer import MeshRenderer

# Load MediaPipe
base_options = python.BaseOptions(model_asset_path="data/mediapipe_models/face_landmarker_v2_with_blendshapes.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options, output_face_blendshapes=True, num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(options)
renderer = MeshRenderer(image_size=256, device="cuda:0")


def extract_blendshapes(obj_path):
    front_img, _ = renderer.render_front_and_back(obj_path)
    # front_img: torch tensor of various shapes
    arr = front_img.detach().cpu().numpy() if hasattr(front_img, 'detach') else np.asarray(front_img)
    print(f"  raw shape: {arr.shape}, dtype: {arr.dtype}")
    # Squeeze batch dim if exists
    while arr.ndim == 4:
        arr = arr.squeeze(0)
    # Transpose CHW → HWC if needed
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)
    # RGBA → RGB
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = (arr * 255 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    result = detector.detect(mp_image)
    if not result.face_blendshapes:
        return None
    blendshapes = result.face_blendshapes[0]
    scores = np.array([b.score for b in blendshapes], dtype=np.float32)
    names = [b.category_name for b in blendshapes]
    return scores, names


# Test trên FaceVerse với expressions khác nhau (cùng identity)
test_files = [
    "/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse/001_01/001_01.obj",  # neutral
    "/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse/001_02/001_02.obj",  # smile
    "/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse/001_03/001_03.obj",  # mouth_stretch
    "/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse/001_04/001_04.obj",  # anger
    "/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse/002_01/002_01.obj",  # different person, neutral
]

print("Testing MediaPipe blendshapes on 5 meshes (4 same-ID different expr, 1 diff-ID):\n")
all_scores = []
for f in test_files:
    if not os.path.exists(f):
        print(f"  {f}: NOT FOUND")
        continue
    result = extract_blendshapes(f)
    if result is None:
        print(f"  {f}: NO FACE DETECTED")
        continue
    scores, names = result
    print(f"\n{os.path.basename(f)}:")
    # Print top-5 active blendshapes
    top_idx = np.argsort(-scores)[:5]
    for idx in top_idx:
        print(f"   {names[idx]:>30s}: {scores[idx]:.3f}")
    all_scores.append((f, scores))

print(f"\n=== Pairwise differences ===")
for i in range(len(all_scores)):
    for j in range(i+1, len(all_scores)):
        diff = np.abs(all_scores[i][1] - all_scores[j][1]).max()
        cs = np.dot(all_scores[i][1], all_scores[j][1]) / (
            np.linalg.norm(all_scores[i][1]) * np.linalg.norm(all_scores[j][1]) + 1e-6)
        print(f"  {os.path.basename(all_scores[i][0])[:15]} vs {os.path.basename(all_scores[j][0])[:15]}: "
              f"max_diff={diff:.3f}, cos_sim={cs:.4f}")

print(f"\n✓ Number of blendshapes: {len(all_scores[0][1]) if all_scores else 0}")
