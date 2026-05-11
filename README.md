# FaceDiff v5 — Quick Start

FaceDiff hiện chạy theo pipeline 2 stage:

1. **Stage 1: SC-VAE** nén O-Voxel sparse features (10 kênh `shape_mat`) thành Slat latents 4096×32.
2. **Stage 2: iMF (Improved Mean Flow)** học dynamics trên Slat và sinh mẫu một bước; backbone mặc định `VoxelMamba`.

Runtime mặc định hiện tại:
- Stage 1: unified `shape_mat` 10 kênh `[dv(3), δ(3), γ(1), rgb(3)]`.
- Stage 2: single-branch iMF với `VoxelMamba` (12 BiMamba layers, hidden 512, d_state 16).
- Dual-branch (shape vs material) vẫn được hỗ trợ qua flag `--dual-branch`.
- Hybrid U-DiT vẫn được hỗ trợ và inference tự nhận backbone theo checkpoint.

## 1. Contract dữ liệu

- Cache chuẩn: `.c10.shape_mat.mx100000.pt`.
- LMDB key payload: `coords`, `features` (10 channels), `aabb`.
- Shape-only legacy mode: `shape_native` 7 kênh.
- Material-only legacy/branch mode: `rgb3` 3 kênh.
- Môi trường chuẩn: `conda activate facediff`.

Thống kê dataset thực tế (đọc từ `resume_contract` trong checkpoint epoch 390):
- FaceVerse: 100 train IDs × 21 expressions = **2,100 mesh**
- FaceScape: 837 train IDs × ~22 mesh trung bình = **18,298 mesh**
- Tổng: **20,398 mesh** (95% train + 5% val ⇒ 19,379 train / 1,019 val)
- LMDB: 20,968 entries (gồm cả test IDs đã cache nhưng filter ra), file `data.mdb` ≈ 429 GB pre-allocated, dữ liệu thực ≈ 272 GB.

## 2. Lệnh train khuyến nghị (unified 10 kênh)

```bash
cd /mnt/18TData/facediff
conda activate facediff

# Stage 1: train-from-scratch SC-VAE (default cosine schedule)
python src/train_sc_vae.py \
  --dataset both \
  --feature-mode shape_mat \
  --in-channels 10 \
  --lmdb-only \
  --checkpoint-dir checkpoints/sc_vae_shape

# Stage 1: resume từ checkpoint epoch 397 với cosine warm-restart
bash scripts/resume_from_397.sh                 # mặc định CKPT=interrupt.pt, EXTEND_EPOCHS=100
EXTEND_EPOCHS=200 bash scripts/resume_from_397.sh   # kéo dài 200 epoch

# Stage 2: iMF mặc định
python src/train_imf.py \
  --dataset both \
  --sc-vae-ckpt checkpoints/sc_vae_shape/latest_step.pt \
  --shape-feature-mode shape_mat \
  --shape-target-in-channels 10
```

Optional: branch-specific experiment

```bash
python src/train_imf.py \
  --dataset both \
  --dual-branch \
  --shape-sc-vae-ckpt checkpoints/sc_vae_shape/latest_step.pt \
  --material-sc-vae-ckpt checkpoints/sc_vae_material/latest_step.pt \
  --shape-feature-mode shape_native \
  --material-feature-mode rgb3 \
  --shape-target-in-channels 7 \
  --material-target-in-channels 3
```

## 3. Sửa lớn so với phiên bản trước (audit ngày 2026-05-10)

Sau khi đối chiếu code với TRELLIS.2 chính thống (`microsoft/TRELLIS.2`,
`trellis2/models/sc_vaes/sparse_unet_vae.py` + `fdg_vae.py`), các điểm sau đã
được canh chỉnh để khớp paper mà vẫn **giữ tương thích checkpoint**
(0 missing/unexpected keys khi `load_state_dict(..., strict=True)`):

1. **Output activations chuẩn TRELLIS.2** — `src/models/sc_vae.py` thêm hàm
   `apply_shape_mat_output_activations()` áp dụng:
   - `dv = (1+2m)·sigmoid(h) − m` với `m=0.5` (paper Eq. Act-1).
   - `δ`: `sigmoid(h)` lúc training (differentiable), `h>0` lúc eval.
   - `γ = softplus(h)` (luôn dương).
   - `rgb = clamp(h, 0, 1)`.
2. **LayerNorm32 (FP32 cast)** trong `SparseResMLPBlock` để chống NaN dưới AMP
   (giống `LayerNorm32` của TRELLIS.2). Vì shape `weight/bias` vẫn giống
   `nn.LayerNorm`, checkpoint cũ load được không sửa state dict.
3. **Non-affine `F.layer_norm`** trước `to_mu/to_logvar` — đúng spec
   `SparseUnetVaeEncoder.forward()` của TRELLIS.2; zero-param ⇒ không phá ckpt.
4. **KL chia `mu.numel()`** thay vì `target_x.shape[0]` — khớp đúng công thức
   báo cáo (Eq. VAE-2). Hệ quả: KL hiển thị chia thêm cho `latent_dim=32`,
   nhưng tổng `kl_weight × kl_loss` ổn định nhờ `kl_weight=1e-6` đã chuẩn theo
   `lambda_kl` của TRELLIS.2.
5. **Render loss dùng activated dv** — `src/scvae_train/render.py` chiếu vị trí
   bằng `apply_dv_activation(recon[..., 0:3])` (paper-spec) thay vì
   `clamp(0,1)`, để supervised tín hiệu khớp với inference path của dual
   contouring.
6. **Resume scheduler extension** — `src/scvae_train/runtime.py:get_resume_scheduler()`
   + CLI flag `--resume-scheduler-mode {continue, constant_min_lr, cosine_restart}`
   cho phép fine-tune sau khi cosine ban đầu đã chạy gần hết (epoch 397/500)
   mà không phải reset optimizer state hay base_lr.

## 4. Debug nhanh cache → mesh

```bash
python scripts/visualize_mesh_vs_ovoxel.py \
  --obj-path <path_obj_goc> \
  --cache-path <path_cache_pt> \
  --out-img outputs/previews/overlay.png \
  --out-recon-obj outputs/previews/recon.obj
```

Ghi chú:
- Dual contouring trong build `o_voxel` hiện phụ thuộc CUDA runtime; khi fail/OOM sẽ fallback marching cubes.
- Nếu GPU đang đầy VRAM, DC fail là hành vi mong đợi chứ không phải pipeline sai.

## 5. Tài liệu nghiên cứu

- `Bao_cao_FaceDiff_ChiTiet.md`: báo cáo chi tiết toán học + kết quả + roadmap.
