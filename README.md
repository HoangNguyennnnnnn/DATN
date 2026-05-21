# FaceDiff v5 — Quick Start

FaceDiff hiện chạy theo pipeline 2 stage:

1. **Stage 1: SC-VAE** nén O-Voxel sparse features (10 kênh `shape_mat`) thành Slat latents 4096×32.
2. **Stage 2: iMF (Improved Mean Flow)** học dynamics trên Slat và sinh mẫu một bước; backbone mặc định `VoxelMamba`.

Runtime mặc định hiện tại:
- Stage 1: unified `shape_mat` 10 kênh `[dv(3), δ(3), γ(1), rgb(3)]`.
- Stage 2: single-branch iMF với `VoxelMamba` (~94M backbone: 12× BiMamba+FFN, dual AdaLN ctx/time, **không prefix**; + v-head ~17M).
- Dual-branch (shape vs material) vẫn được hỗ trợ qua flag `--dual-branch`.
- Hybrid U-DiT vẫn được hỗ trợ và inference tự nhận backbone theo checkpoint.

### Cài `o_voxel` (không có trên PyPI)

Gói PyPI `o-voxel` không tồn tại; FaceDiff dùng **`o_voxel`** từ [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (`subdirectory=o-voxel`). Sau khi `pip install -r requirements.txt` **thành công**, chạy:

```bash
apt-get update && apt-get install -y git build-essential cmake ninja-build
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export MAX_JOBS=4
bash scripts/install_o_voxel.sh
```

Script cài lần lượt **CuMesh → FlexGEMM → o_voxel** (cùng repo TRELLIS.2). `CUDA_HOME` phải trùng toolkit đang dùng để compile (`nvcc --version`).

Nếu báo **không có nvcc**: image chỉ có runtime GPU — cài compiler ví dụ `apt-get install -y cuda-nvcc-12-4` (đổi bản cho khớp repo NVIDIA / driver), hoặc dùng Docker **`nvidia/cuda:*-devel`**.

**Nếu build `cumesh` / `flex_gemm` / `o_voxel` vẫn fail:** xem log đầy đủ (`pip install ... -v`). Trên **server chỉ resume train từ LMDB đã pack**, có thể **không cần** `o_voxel`: dùng `--lmdb-dir ... --lmdb-only` cho SC-VAE và `--ovoxel-lmdb` cho precompute — không gọi mesh→O-Voxel trên máy đó. Khi đó chỉ cần giải nén data LMDB + checkpoint; `o_voxel` chỉ bắt buộc nếu bạn chạy pipeline đọc `.obj` và convert.

### Mamba-ssm / causal-conv1d (tùy chọn — hay lỗi `Failed building wheel`)

`requirements.txt` **không** còn cài `mamba-ssm` / `causal-conv1d` (phải compile CUDA). **VoxelMamba** tự **fallback GRU** nếu chưa có hai gói này — train vẫn chạy, chỉ chậm hơn.

Để bật kernel Mamba, sau khi `torch` + `nvcc` OK:

```bash
apt-get update && apt-get install -y ninja-build git build-essential cmake
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export MAX_JOBS=4
bash scripts/install_mamba_optional.sh
```

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

# Stage 1: resume với EMA + normal loss + cosine restart
python src/train_sc_vae.py \
  --resume checkpoints/sc_vae_shape/interrupt.pt \
  --lr 1e-5 --no-torch-compile \
  --batch-size 1 --gradient-accumulation-steps 132 \
  --enable-ema --ema-decay 0.9999 \
  --enable-stage2-render-loss \
  --resume-scheduler-mode cosine_restart \
  --resume-extend-epochs 200 --resume-target-min-lr 1e-6

# Hoặc dùng script tự động (Steps 4→5→6)
bash scripts/resume_from_step4.sh

# Stage 2: iMF (khuyến nghị — LMDB + CFG off)
bash scripts/train_imf.sh
# Hoặc thủ công:
python src/train_imf.py \
  --offline-data \
  --slat-lmdb data/slat_context.lmdb \
  --context-lmdb data/hybrid_context.lmdb \
  --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
  --batch-size 4 --gradient-accumulation-steps 16 \
  --disable-cfg-conditioning --disable-id-filters
```

Precompute (một lần) trước khi pack LMDB:

```bash
python scripts/precompute_slat_cache.py \
  --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
  --dataset both --context-lmdb data/hybrid_context.lmdb --skip-existing
python scripts/pack_slat_lmdb.py
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

## 3. Sửa lớn so với phiên bản trước (audit ngày 2026-05-10, cập nhật 2026-05-14)

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
7. **EMA (Exponential Moving Average)** — `src/scvae_train/runtime.py:EMA` class,
   decay=0.9999 theo TRELLIS.2 standard. Shadow weights lưu trên GPU (~134MB cho 35M params).
   Tích hợp vào training loop (update sau optimizer step), validation (apply_shadow/restore),
   checkpoint (save/load `ema_state_dict`). CLI: `--enable-ema`, `--ema-decay`.
8. **Depth-to-Normal Loss** — `src/scvae_train/render.py:_depth_to_normal()` tính
   pháp tuyến bề mặt từ depth maps bằng sai phân hữu hạn (finite differences).
   Tích hợp vào Stage 2 render loss cho tất cả feature modes (shape_mat, geom6,
   shape_native, geom_mat12) với λ_normal=1, khớp TRELLIS.2 Eq.8.
   Loss: L1(normal) + L1 + 0.2×SSIM + 0.2×LPIPS trên normal maps.
9. **nvdiffrast 0.4.0** — Đã cài đặt NVIDIA differentiable rasterizer.
   Hiện chưa tích hợp vào render loss (vẫn dùng point projection);
   sẽ thay thế bằng mesh rasterization ở lượt train tiếp theo.

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

## 5. Stage 2 — thay đổi kiến trúc (2026-05-22, commit `c92ba00`)

| Trước | Hiện tại |
|-------|----------|
| 24 prefix tokens (ctx+t+r+interval+guidance) | **0 prefix** — `mamba_num_*_tokens=0` |
| `cond_fusion` concat ctx+time | **Dual AdaLN**: `context_cond_mlp` + `time_guidance_mlp` |
| Chỉ BiMamba, không FFN | **FFN** mỗi block (DiT-style) + gated residual |
| `output_proj` zero-init | **Xavier gain=0.02** (tránh chết gradient backbone) |
| AdaLN gate bias=0 | **gate bias=1.0** |
| CFG bật mặc định | **`--disable-cfg-conditioning`** trong `train_imf.sh` |
| ~21M params | **~94M** backbone + **~17M** v-head + contrastive |

Checkpoint cũ (`adaLN_modulation`, `cond_fusion`, prefix) **không tương thích** — train lại từ scratch.

Diagnostic scripts: `scripts/test_pure_t1.py` (memorization 1 mẫu), `scripts/test_imf_identity_t0.py`, `scripts/test_imf_memorization.py`.

## 6. Tài liệu nghiên cứu

- `Bao_cao_FaceDiff_ChiTiet.md`: báo cáo chi tiết toán học + kết quả + roadmap.
- `CLAUDE.md`: tóm tắt kiến trúc + lệnh thường dùng cho agent.
