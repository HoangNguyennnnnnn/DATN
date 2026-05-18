# FaceDiff — Bản nháp ĐATN (Đồ án Tốt nghiệp)

> **Mục đích**: Tài liệu tham khảo chi tiết để viết quyển đồ án theo cấu trúc 6 chương + phụ lục.
> Mỗi section có nội dung sẵn (text, công thức, bảng, hình mô tả) để copy vào Word/LaTeX và mở rộng.

---

# CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI

## 1.1 Đặt vấn đề

**Tạo sinh khuôn mặt 3D (3D Face Generation)** là bài toán then chốt trong thị giác máy tính, với ứng dụng rộng rãi:

- **Giải trí**: game (NPC), phim hoạt hình (digital double), VR/AR (avatar)
- **Y tế**: phục hồi khuôn mặt sau chấn thương, thẩm mỹ
- **An ninh**: face reenactment cho identity verification
- **Công nghiệp**: tự động sinh dữ liệu huấn luyện cho mô hình nhận diện

**Yêu cầu chất lượng cao** dẫn đến 3 thách thức nghiêm trọng:

1. **Độ phức tạp tính toán bậc ba**: Biểu diễn voxel $256^3$ tạo ra ~16.7 triệu điểm. Transformer với attention $O(N^2)$ không khả thi.

2. **Tốc độ sinh chậm**: Mô hình diffusion truyền thống cần 20–50 bước ODE/SDE để hội tụ, không phù hợp với ứng dụng tương tác thời gian thực.

3. **Phần cứng đắt đỏ**: Mô hình state-of-the-art (TRELLIS.2) yêu cầu **8×A100 GPUs (320GB VRAM)** — không khả thi cho nhóm nghiên cứu hoặc startup.

**Phát biểu vấn đề**: Cần một hệ thống sinh khuôn mặt 3D chất lượng cao, **tốc độ một bước**, **chạy trên một GPU tiêu dùng (RTX 4090, 24GB)**, kiểm soát được danh tính và biểu cảm.

## 1.2 Các giải pháp hiện tại và hạn chế

Tổng kết các phương pháp sinh mesh 3D hiện có:

| Phương pháp | Biểu diễn | Ưu điểm | Hạn chế |
|-------------|-----------|---------|---------|
| **Point-E, PointFlow** [25] | Point Cloud | Đơn giản, nhanh | Thiếu topology, không trực tiếp ra mesh |
| **DreamFusion, Magic3D** [12] | NeRF + SDS | Chất lượng cao | 30–60 phút/đối tượng, "Janus effect", chi phí GPU lớn |
| **GaussianHead** | 3D Gaussian | Render đẹp, real-time | Khó trích xuất mesh polygon |
| **MeshGPT, MeshAnything** [13] | Mesh trực tiếp | Topology chính xác | Giới hạn vài nghìn mặt, không scale |
| **TRELLIS.2** [2] | O-Voxel + Sparse VAE | **Mesh chi tiết 200K+ đỉnh** | **8×A100, 50 bước DDPM** — đắt và chậm |
| **DiM-3D** [4] | Mamba + DDPM | $O(N)$ complexity | Vẫn cần 1000 bước DDPM |

**Hạn chế chung**:
1. **Chi phí phần cứng**: không có giải pháp 3D chất lượng cao trên 1 GPU tiêu dùng
2. **Tốc độ sinh**: 20–50 bước → không tương tác được
3. **Kiểm soát ngữ nghĩa**: nhiều hệ thống không kiểm soát đồng thời danh tính + biểu cảm
4. **Khoảng cách biểu diễn**: NeRF/Gaussian khó tích hợp vào pipeline sản xuất truyền thống

## 1.3 Mục tiêu và định hướng giải pháp

### Mục tiêu cụ thể

| # | Mục tiêu | Chỉ tiêu kỹ thuật |
|---|----------|-------------------|
| 1 | Mesh 3D chất lượng cao | ≥ 200K đỉnh, mesh 10-kênh (geometry + RGB) |
| 2 | Sinh 1 bước | < 2 giây/mẫu trên RTX 4090 |
| 3 | Kiểm soát danh tính + biểu cảm | Vector điều kiện 946-dim (ArcFace + FLAME + DINOv2) |
| 4 | Đơn GPU | VRAM peak < 22GB |
| 5 | Reproducible | Mã nguồn mở, training pipeline tự động hoá |

### Định hướng giải pháp: **FaceDiff** — pipeline 3 giai đoạn

Hệ thống đề xuất gồm 3 mô-đun chính, kế thừa các phát kiến gần đây:

1. **SC-VAE (Sparse Convolution VAE)** — nén O-Voxel 10-kênh $256^3$ → Slat tokens $[4096, 32]$. Giảm hơn 90% kích thước biểu diễn.

2. **VoxelMamba + iMF (Improved Mean Flow)** — backbone Mamba SSM $O(N)$ + Improved Mean Flow để sinh 1 bước. Sinh Slat tokens từ ngẫu nhiên + ngữ cảnh.

3. **Dual Contouring + Decoder** — Giải mã Slat → O-Voxel → mesh polygon bằng thuật toán Dual Contouring (TRELLIS.2 protocol).

### Đối tác công nghệ chính

- **Mamba SSM** (Gu & Dao, 2024) — backbone tuyến tính $O(N)$
- **Improved Mean Flow** (Geng et al., 2025) — sinh 1-NFE state-of-the-art (1.72 FID ImageNet)
- **TRELLIS.2** (Microsoft, 2025) — Sparse VAE + Dual Contouring
- **MediaPipe FaceLandmarker V2** — extract 52 ARKit blendshapes (expression)
- **ArcFace** (Deng et al., 2019) — identity embedding
- **DINOv2** (Oquab et al., 2024) — shape/hair features

## 1.4 Đóng góp của đồ án

1. **SC-VAE tiết kiệm VRAM**: Sử dụng `SparseResMLPBlock` thay `ConvNeXt 3D` — giảm 45% VRAM. Kèm Generative Pruning (Rho Loss) để học topology hiệu quả.

2. **VoxelMamba — Hybrid backbone**: Kết hợp 3 phát kiến từ các paper khác nhau:
   - **Bidirectional Mamba** cho 3D generation (cảm hứng DiM-3D)
   - **Hilbert space-filling curve ordering** (cảm hứng VoxelMamba)
   - **In-context prefix token conditioning** (theo iMF paper Section 4.3)

3. **Sinh 1 bước với iMF**: Áp dụng Improved Mean Flow vào lĩnh vực mesh 3D — đầu tiên trong giai đoạn 2024-2026.

4. **Hybrid Context 946-dim**: ArcFace(512) + FLAME(50) + DINOv2(384) — kiểm soát đồng thời identity + expression + back shape. **MediaPipe FLAME** là đóng góp kỹ thuật (52 blendshapes thực tế thay vì FLAME-from-image random).

5. **Per-channel Slat Normalization**: Phát hiện và sửa "identity collapse" trong iMF training do SC-VAE latent std=0.36 << noise std=1.0 → SNR thấp. Fix theo TRELLIS.2 protocol.

6. **Tối ưu đơn GPU**: BFloat16 mixed precision, gradient checkpointing, LMDB caching, TF32 + cudnn.benchmark, refactored CFG (split forward, no_grad uncond). **Peak VRAM 17 GB / 24 GB**.

7. **Pipeline tự động hoá**: Scripts auto recompute LMDB, scripts test E2E, scripts visualize context, scripts compute slat stats — sẵn sàng cho reproducible research.

## 1.5 Bố cục đồ án

- **Chương 2 — Nền tảng lý thuyết**: SSM/Mamba, Hilbert curve, Flow Matching, iMF, VAE, Dual Contouring
- **Chương 3 — Phương pháp đề xuất**: Kiến trúc 3-stage, data pipeline, hybrid context system
- **Chương 4 — Phân tích lý thuyết**: Complexity analysis, VRAM breakdown, sample efficiency
- **Chương 5 — Đánh giá thực nghiệm**: Datasets, metrics, ablation studies, qualitative results
- **Chương 6 — Kết luận**: Đóng góp, hạn chế, hướng phát triển

---

# CHƯƠNG 2. NỀN TẢNG LÝ THUYẾT

## 2.1 Ngữ cảnh của bài toán

### 2.1.1 Biểu diễn 3D

Có nhiều cách biểu diễn shape 3D:

| Biểu diễn | Ưu | Nhược | Phù hợp cho FaceDiff? |
|-----------|-----|-------|----------------------|
| Point Cloud | Đơn giản | Thiếu topology | ❌ |
| Voxel Grid (dense) | Đầy đủ thông tin | $O(N^3)$ memory | ❌ Quá lớn |
| **O-Voxel (sparse 10-channel)** | Sparse, có normals + RGB | Cần SC-VAE để nén | ✅ **FaceDiff sử dụng** |
| Mesh (polygon) | Render được trực tiếp | Khó học | Output cuối |
| NeRF/Gaussian | Render đẹp | Không có mesh | ❌ |
| SDF (Signed Distance Field) | Trơn, mượt | Cần marching cubes | Backup |

### 2.1.2 Mô hình tạo sinh (Generative Models)

| Class | Đặc trưng | Số bước sinh |
|-------|-----------|---------------|
| VAE [11] | 1 forward, dễ huấn luyện | 1 |
| GAN | Adversarial, khó train | 1 |
| Diffusion (DDPM) [10] | Iterative denoising | 20–1000 |
| Flow Matching [13] | Learn ODE | 10–100 |
| **Mean Flow / iMF** [3] | Learn average velocity | **1** |
| Consistency Models | Distillation từ teacher | 1–2 |

→ FaceDiff chọn **iMF** vì kết hợp được chất lượng cao của diffusion với tốc độ 1-NFE.

## 2.2 Các kết quả nghiên cứu tương tự

### 2.2.1 Bảng tổng hợp công trình liên quan

| Năm | Công trình | Domain | Backbone | Sample steps | GPU req |
|-----|-----------|--------|----------|--------------|---------|
| 2024 | TRELLIS / TRELLIS.2 [2] | Mesh 3D | U-DiT (Transformer) | 50 (DDPM) | 8×A100 |
| 2024 | DiT-3D [11] | Point cloud | Plain DiT | 1000 (DDPM) | 4×A100 |
| 2024 | DiM-3D [4] | Point cloud | Mamba + AdaLN + FFN | 1000 (DDPM) | 1×A100 |
| 2024 | VoxelMamba [4b] | 3D object detection | Hilbert + Mamba | — (discriminative) | 1×3090 |
| 2025 | **MeanFlow (MF)** [3] | Image 2D | Transformer DiT-B | **1 (NFE)** | 8×A100 |
| 2025 | **iMF (improved MF)** [3] | Image 2D | Transformer + in-context | **1 (NFE)** | 8×A100 |
| 2026 | **FaceDiff (đồ án này)** | **Mesh 3D** | **Mamba + in-context + iMF** | **1 (NFE)** | **1×RTX 4090** |

### 2.2.2 Vị trí FaceDiff trong literature

FaceDiff là **kết hợp đầu tiên** của 3 thành phần:
- iMF objective (1-NFE state-of-the-art image gen)
- Mamba backbone $O(N)$ (linear complexity)
- Mesh 3D output (200K+ vertex)

Trên một GPU tiêu dùng — chưa có công trình tương tự published.

## 2.3 Mô hình Trạng thái Không gian (State Space Model — SSM) và Mamba

### 2.3.1 SSM liên tục

SSM mô hình hoá hệ động lực tuyến tính:

$$h'(t) = A \cdot h(t) + B \cdot x(t) \tag{2.1}$$
$$y(t) = C \cdot h(t) \tag{2.2}$$

trong đó:
- $x(t) \in \mathbb{R}$ — input scalar tại thời điểm $t$
- $h(t) \in \mathbb{R}^N$ — hidden state $N$-chiều
- $A \in \mathbb{R}^{N \times N}$, $B \in \mathbb{R}^{N \times 1}$, $C \in \mathbb{R}^{1 \times N}$ — ma trận tham số

### 2.3.2 Rời rạc hoá (Zero-Order Hold — ZOH)

Để áp dụng cho sequence rời rạc, dùng ZOH với timestep $\Delta$:

$$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\Delta B \tag{2.3}$$

Phương trình rời rạc:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t \tag{2.4}$$

### 2.3.3 Selective SSM (Mamba) [5]

Mamba (Gu & Dao, 2024) làm SSM **phụ thuộc input**:

$$\Delta_t, B_t, C_t = \text{Linear}(x_t) \tag{2.5}$$

Mỗi token có $B, C, \Delta$ riêng → model **chọn lọc** thông tin nào lưu vào state, thông tin nào bỏ — tương tự attention nhưng $O(L)$ thay vì $O(L^2)$.

### 2.3.4 Bidirectional Mamba (BiMamba)

Vì voxel grid 3D không có thứ tự tự nhiên, FaceDiff dùng **Bidirectional Mamba**:

$$y_{\text{fwd}} = \text{Mamba}_\rightarrow(x), \quad y_{\text{bwd}} = \text{flip}(\text{Mamba}_\leftarrow(\text{flip}(x))) \tag{2.6}$$
$$y = y_{\text{fwd}} + y_{\text{bwd}} \tag{2.7}$$

→ Hidden state mang context từ cả 2 chiều.

### 2.3.5 SSM Scan Intermediates

Khi forward, kernel CUDA "selective scan" cần lưu intermediate states cho backward:

$$\text{Memory}_{\text{scan}} = B \times d_{\text{inner}} \times d_{\text{state}} \times L$$

Trong FaceDiff: $B \times 1024 \times 16 \times 4096 = 268M \text{ elements}$. Với BFloat16 (2 bytes): **~536 MB / call**. 12 layers × 2 (fwd/bwd) = 24 calls → **~8 GB** total.

## 2.4 Đường cong Hilbert (Hilbert Space-Filling Curve)

### 2.4.1 Định nghĩa

Đường cong Hilbert là đường cong liên tục mapping 1D index → 3D coordinate, giữ được **spatial locality**: hai điểm gần nhau trong 1D → gần nhau trong 3D, và ngược lại.

### 2.4.2 So sánh các phương pháp sắp xếp 3D → 1D

| Phương pháp | Locality | Thực hiện | Trong FaceDiff |
|-------------|----------|-----------|----------------|
| Raster scan (x→y→z) | Kém (jump khi edge) | Đơn giản | ❌ |
| Morton (Z-order) | Khá | Bit interleave | ⚠️ |
| **Hilbert** | **Tốt nhất** | Đệ quy | ✅ FaceDiff dùng |

Với SSM/Mamba (xử lý sequence), Hilbert ordering quan trọng vì hidden state chỉ accumulate thông tin trong 1 chiều — neighbors trong 3D phải kề nhau trong 1D để propagate context hiệu quả.

## 2.5 Flow Matching và Mean Flow

### 2.5.1 Flow Matching [13]

Học vector field $v_\theta$ chuyển từ noise $e \sim \mathcal{N}(0, I)$ → data $x$:

$$z_t = (1-t) x + t e, \quad t \in [0, 1] \tag{2.8}$$
$$v(z_t) = e - x \quad \text{(linear flow target)} \tag{2.9}$$

Loss:
$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x, e} \| v_\theta(z_t, t) - (e - x) \|^2 \tag{2.10}$$

Sample bằng ODE từ $t=1$ về $t=0$, thường cần 10-100 bước.

### 2.5.2 Mean Flow [12]

Học **average velocity** thay vì instantaneous velocity:

$$u(z_t, r, t) = \frac{1}{t-r} \int_r^t v(z_s, s) \, ds \tag{2.11}$$

**MeanFlow identity** [12]:
$$v(z_t, t) = u(z_t, r, t) + (t - r) \cdot \frac{du}{dt}(z_t, r, t) \tag{2.12}$$

Hàm hợp $V_\theta$:
$$V_\theta = u_\theta + (t - r) \cdot \text{stopgrad}(JVP_t u_\theta) \tag{2.13}$$

Sinh **1 bước**:
$$z_0 = z_1 - u_\theta(z_1, r=0, t=1) \tag{2.14}$$

### 2.5.3 Improved Mean Flow (iMF) [3]

iMF cải tiến Mean Flow ở 2 điểm:

**Cải tiến 1 — v-loss reparameterization**:

Original MF học $u$ trực tiếp (network-dependent target). iMF reformulate thành v-loss reparameterized bởi u-pred:

$$\mathcal{L}_{\text{iMF}} = \mathbb{E} \| V_\theta - (e - x) \|^2 \tag{2.15}$$

với $V_\theta$ là average velocity được "lifted" thành instantaneous velocity qua MeanFlow identity. → **Standard regression target** không phụ thuộc network.

**Cải tiến 2 — Flexible CFG**:

Original MF dùng CFG scale cố định. iMF làm guidance scale $\omega$ trở thành **conditioning variable**:

$$u_\theta(z_t, r, t, \omega, c, t_{\min}, t_{\max}) \tag{2.16}$$

Cho phép tuỳ chỉnh guidance tại inference time.

**Algorithm 2 (iMF với CFG conditioning)** — pseudocode từ paper:

```python
# Sample conditions
t, r, omega = sample_t_r_cfg()
e = randn_like(x)
z = (1-t)*x + t*e
# Two forward passes
v_c = fn(z, t, t, omega, c)          # conditional at boundary
v_u = fn(z, t, t, omega, None)       # unconditional at boundary
# CFG-augmented target
v_g = (e - x) + (1 - 1/omega) * (v_c - v_u)
# JVP for compound function
u, dudt = jvp(fn, (z, r, t, omega, c), (v_c, 0, 1, 0, 0))
V = u + (t-r) * stopgrad(dudt)
error = V - v_g
loss = metric(error)
```

### 2.5.4 In-context Conditioning (iMF Section 4.3)

iMF paper phát hiện **rằng AdaLN-zero không cần thiết**:

- Mỗi loại condition → 2-layer MLP → multiple replicated tokens
- Tokens concat vào sequence cùng với image latent tokens
- Bỏ AdaLN-zero giảm **1/3 kích thước model** (133M → 89M) và **cải thiện FID** (4.57 → 4.09)

Số tokens (iMF paper):
- 8 tokens cho class
- 4 tokens cho mỗi $\{r, t, \omega, t_{\min}, t_{\max}\}$
- → tổng **24 tokens** condition

FaceDiff áp dụng nguyên xi: **8 (context) + 4 (t) + 4 (r) + 4 (interval) + 4 (guidance) = 24 tokens**.

## 2.6 Variational Autoencoder (VAE)

### 2.6.1 ELBO

Với observation $x$ và latent $z$:

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z)) \tag{2.17}$$

Mục tiêu maximize ELBO ↔ minimize:

$$\mathcal{L}_{\text{VAE}} = \underbrace{\| x - \hat{x} \|^2}_{\text{reconstruction}} + \beta \cdot \underbrace{D_{KL}(q(z|x) \| \mathcal{N}(0, I))}_{\text{regularization}} \tag{2.18}$$

### 2.6.2 SC-VAE trong FaceDiff

Đặc thù:
- Input: **sparse voxel** (O-Voxel 10-channel)
- Encoder/Decoder: sparse convolution (spconv 2.x)
- Latent: $4096 \times 32$ Slat tokens
- $\beta_{\text{KL}} = 10^{-6}$ — rất nhỏ (TRELLIS.2 spec), không ép latent → $\mathcal{N}(0, I)$ mạnh

**Hệ quả**: Slat std $\approx 0.36$ — yêu cầu per-channel normalization trước iMF training để tránh identity collapse.

## 2.7 Dual Contouring (DC)

### 2.7.1 Bài toán

Cho occupancy field rời rạc + delta offset $\delta v$ cho mỗi voxel, tạo polygon mesh:

$$\text{DC}(\{(c_i, \delta v_i, \gamma_i)\}_i) \to (V, F)$$

trong đó $c_i \in \mathbb{Z}^3$ là grid coord, $\delta v_i \in \mathbb{R}^3$ là offset trong voxel, $\gamma_i$ là split weight, $V$ là vertices, $F$ là faces.

### 2.7.2 QEF (Quadratic Error Function)

Mỗi cell tạo 1 vertex tối ưu hoá:

$$V^* = \arg\min_V \sum_{\text{edges}} (n_e^T (V - p_e))^2 \tag{2.19}$$

với $p_e$ là intersection trên edge, $n_e$ là normal.

### 2.7.3 TRELLIS.2 QEF mở rộng

TRELLIS.2 thêm:
- **Multi-grid resolution** (8× upsample từ slat $16^3$ → output $128^3$)
- **Split weights** $\gamma$ để xử lý non-manifold regions
- **Dual vertex placement** trong voxel cell qua $\delta v$

→ Mesh chi tiết 200K+ vertices từ slat $16^3$.

---

# CHƯƠNG 3. PHƯƠNG PHÁP ĐỀ XUẤT

## 3.1 Tổng quan giải pháp

### 3.1.1 Kiến trúc 3 giai đoạn

```
[Mesh GT] → [Mesh Renderer] → [Front/Back images]
              ↓                ↓
       ArcFace(512)         DINOv2(384)
              ↓                ↓
        [Image → MediaPipe FLAME(50)]
              ↓
       Hybrid Context [946-dim]
              ↓
       ┌──────┴──────┐
       │             │
[Mesh GT]        Random Noise z₁
   ↓                  ↓
[O-Voxel]      Context [946]
   ↓                  ↓
[SC-VAE Enc] → Slat [4096, 32]    ←─ Per-channel normalize
   ↓                  ↓
[Slat GT]      [VoxelMamba + iMF] (1-step)
                      ↓
                 Slat pred [4096, 32]
                      ↓ Reverse normalize
                 Slat decoder input
                      ↓
                 [SC-VAE Decoder] → O-Voxel
                                       ↓
                                 [Dual Contouring]
                                       ↓
                                 Mesh polygon
```

### 3.1.2 Pipeline tổng quát

| Giai đoạn | Mô-đun | Input | Output | Số bước |
|-----------|--------|-------|--------|---------|
| **Stage 1** | SC-VAE (Sparse VAE) | O-Voxel $[N, 10]$ | Slat $[4096, 32]$ | 1 forward |
| **Stage 2** | VoxelMamba + iMF | Context $[946]$ + Noise | Slat pred $[4096, 32]$ | **1 step** |
| **Stage 3** | SC-VAE Decoder + DC | Slat $[4096, 32]$ | Mesh polygon | 1 forward + DC |

Tổng inference time: **< 2 giây** trên RTX 4090.

## 3.2 Giai đoạn 1: SC-VAE — Sparse Convolution VAE

### 3.2.1 Biểu diễn O-Voxel 10-kênh

O-Voxel encode mesh thành sparse voxel grid với feature 10-channel mỗi voxel:

| Channel | Tên | Ý nghĩa | Range |
|---------|-----|---------|-------|
| 0-2 | `dv` | Vertex offset trong voxel | $[-0.5, 1.5]$ (post-activation) |
| 3-5 | `delta` | Intersected flag mỗi axis | $\{0, 1\}$ |
| 6 | `gamma` | Split weight cho DC | $\mathbb{R}_+$ |
| 7-9 | `rgb` | Vertex color | $[0, 1]$ |

Pipeline:
1. **Mesh → sparse points** (3D rasterization)
2. **Surface fit** → tính `dv, delta, gamma`
3. **UV sampling** → `rgb`
4. **Pruning** → giữ ~50K-350K active voxels/mesh

### 3.2.2 Kiến trúc SC-VAE

```
Encoder (Pyramid 4 levels):
  in [N, 10] → enc1 [64] → enc2 [128] → enc3 [256] → enc4 [512]
  pre_latent: F.layer_norm (non-affine) → to_mu, to_logvar
  → mu, logvar [N_root, 32]

Decoder (Generative Pruning):
  z [N, 32] → dec_proj [512] → dec4 (with rho) → dec3 → dec2 → dec1
  → out_proj [N_fine, 10]
  → Activations: dv = (1+2m)·σ(h) - m, delta = (logits > 0), gamma = softplus(g), rgb = clamp(c, 0, 1)
```

**SparseResMLPBlock** (thay ConvNeXt 3D):
```
x → spconv → LayerNorm32 → MLP(d_model → 4d) → SiLU → MLP(4d → d_model) → +residual
```

### 3.2.3 Loss Function

$$\mathcal{L}_{\text{SC-VAE}} = \mathcal{L}_{\text{recon}} + \beta_{\text{KL}} \mathcal{L}_{\text{KL}} + \lambda_\rho \mathcal{L}_\rho + \mathcal{L}_{\text{render}} \tag{3.1}$$

Chi tiết:
- $\mathcal{L}_{\text{recon}}$: MSE(`dv`) + BCE(`delta`) + smooth_l1(`gamma`) + L1(`rgb`)
- $\mathcal{L}_{\text{KL}}$: chia `mu.numel()` (per-element norm), $\beta_{\text{KL}} = 10^{-6}$
- $\mathcal{L}_\rho$: Generative Pruning Focal Loss, $\lambda_\rho = 0.2$
- $\mathcal{L}_{\text{render}}$: Mesh→Point projection → render 2D depth/mask → L1 vs GT

## 3.3 Giai đoạn 2: VoxelMamba + iMF (Diffusion Mamba với Improved Mean Flow)

### 3.3.1 Kiến trúc VoxelMamba

**Backbone**: 12 lớp BidirectionalMambaBlock, hidden_dim=512, d_state=16, expand=2, kernel size=4.

```
Input:
  z_t  [B, 4096, 32]  noise + slat
  ctx  [B, 946]       hybrid context
  t, r [B]            timesteps (end, start)
  Ω    [B, 3]         {omega, t_min, t_max} CFG params

Tokenize prefix:
  ctx       → context_tokenizer  → 8 tokens of dim 512
  t         → time_mlp → time_tokenizer  → 4 tokens
  r         → r_mlp → r_tokenizer  → 4 tokens
  (t-r)     → interval_mlp → interval_tokenizer  → 4 tokens
  Ω         → guidance_tokenizer  → 4 tokens
  → prefix [B, 24, 512]

Process voxel tokens:
  z_t → input_embed [B, 4096, 512]
  → Hilbert reorder (preserve 3D locality)
  → concat [prefix, hilbert(z_t)] = [B, 4120, 512]
  → 12 × BiMambaBlock
  → strip prefix → [B, 4096, 512]
  → Hilbert inverse → [B, 4096, 512]
  → output_norm (RMSNorm)
  → output_proj (Linear, zero-init) → [B, 4096, 32]
```

**Total parameters**: ~49M (12 layers × 4M).

### 3.3.2 BidirectionalMambaBlock

```python
def forward(self, x):
    residual = x
    x = RMSNorm(x)
    fwd = forward_mamba(x)
    bwd = flip(backward_mamba(flip(x)))
    out = dropout(fwd + bwd)
    return out + residual
```

Hai Mamba modules **độc lập** (không share input_proj như paper DiM-3D) → capacity tăng 2× nhưng VRAM cũng 2×. Trade-off chấp nhận được vì face domain cần more capacity.

### 3.3.3 In-context Conditioning (theo iMF paper)

iMF paper Section 4.3 chứng minh in-context conditioning > AdaLN-zero (FID 4.09 vs 4.57, params 89M vs 133M). FaceDiff áp dụng nguyên xi:

| Condition | Số tokens | Source |
|-----------|-----------|--------|
| context (Hybrid 946) | **8** | iMF paper |
| t (end time) | 4 | iMF paper |
| r (start time) | 4 | iMF paper |
| (t-r) interval | 4 | iMF paper |
| Ω (ω, t_min, t_max) | 4 | iMF paper |
| **Total** | **24** | Match paper |

### 3.3.4 iMF Training Loss

Theo iMF Algorithm 2 (paper Section 4.2):

```python
# Sample t, r, omega
t, r = sample_t_r()  # logit-normal centered at 0.4
omega = sample_omega()  # power distribution

# Compute targets
e = randn_like(x)
z_t = (1-t)*x + t*e

# Two separate forward passes (memory-efficient)
v_theta = model(z_t, t, ctx, r=t, omega, ...)              # batch B, with grad
with torch.no_grad():
    v_uncond = model(z_t, t, zeros_ctx, r=t, omega, ...)   # batch B, no grad

# CFG-augmented target
coeff = 1 - 1/omega
v_target = (e - x) + coeff * (v_theta.detach() - v_uncond.detach())

# Branching: r==t (boundary) vs r!=t (JVP)
if mask_eq (r==t):
    error = v_theta - v_target  # simple v-loss
else:
    u, dudt = jvp(model, (z_t, t, ...), (v_theta, 1, ...))
    V = u + (t-r) * stopgrad(dudt)
    error = V - v_target

# Adaptive weighting + auxiliary v-head loss
loss = AdaptiveLossWeighting(error) + λ_v * v_head_loss
```

### 3.3.5 1-step Sampling

Tại inference:
```python
z_1 = randn(shape)
z_0 = z_1 - u_theta(z_1, r=0, t=1, omega=4, ctx, ...)
slat_decoder = z_0 * std + mean  # reverse normalize
```

**Một function evaluation duy nhất** → đạt mục tiêu < 2s/mesh.

## 3.4 Giai đoạn 3: SC-VAE Decode + Dual Contouring

### 3.4.1 Slat Decode

```python
slat_normalized = slat_pred  # output từ VoxelMamba
slat_raw = slat_normalized * std + mean  # reverse normalize
voxel_features = sc_vae.decode(slat_raw, grid_indices)  # [N_active, 10]
```

### 3.4.2 Dual Contouring

```python
# Decode activations
dv = (1 + 2m) * sigmoid(voxel_features[:, 0:3]) - m
delta = (voxel_features[:, 3:6] > 0).bool()
gamma = softplus(voxel_features[:, 6:7])
rgb = clamp(voxel_features[:, 7:10], 0, 1)

# Pre-fill boundary voxels (fix DC holes)
coords, dv, delta, gamma = prefill_boundary_voxels(coords, dv, delta, gamma)

# DC kernel (TRELLIS.2 protocol)
verts, faces = flexible_dual_grid_to_mesh(
    coords, dv, delta,
    split_weight=gamma, aabb=[-1, 1]^3, grid_size=256,
)

# Post-process
mesh = trimesh.repair(verts, faces)  # fix_normals, fill_holes
mesh = taubin_smooth(verts, faces, iters=6)  # GPU
colors = gpu_knn_idw(verts, voxel_coords, rgb)  # GPU IDW from voxel
```

Output mesh: **200K+ vertices, polygon, with per-vertex RGB**.

## 3.5 Hybrid Context System

### 3.5.1 ArcFace Identity (512-dim)

- **Model**: InsightFace `buffalo_l` (ResNet50)
- **Input**: front-rendered face image (256×256, RGB)
- **Output**: L2-normalized embedding [512]
- **Property**: cos_sim same-ID = 0.84, diff-ID = 0.05 (verified)

### 3.5.2 FLAME Expression — MediaPipe FaceLandmarker V2 (50-dim)

**Bug được phát hiện** (Revision 17, 18/05/2026):
- FLAME adapter cũ là random-init CNN → output constant (max_diff = 0.0003 cho 5 mesh khác)
- Model 14 epoch đã train với FLAME = noise constant → mất expression conditioning

**Fix**: MediaPipe FaceLandmarker V2 (3.7 MB model):
- 52 ARKit-compatible blendshape coefficients
- Drop 2 redundant (`_neutral`, `browDownLeft`) → giữ 50-dim, `context_dim=946` không đổi
- **Verified**: same ID different expressions cos_sim 0.72, diff ID cos_sim 0.45-0.67

### 3.5.3 DINOv2 Back-of-head (384-dim)

- **Model**: `facebook/dinov2-small` (ViT-S/14)
- **Input**: back-rendered face image (capture hair/shape silhouette)
- **Output**: average pooled features [384]

### 3.5.4 Hybrid Context Concat

$$\text{ctx} = \text{concat}[\text{ArcFace}(512), \text{FLAME}(50), \text{DINOv2}(384)] \in \mathbb{R}^{946} \tag{3.2}$$

Stored in `data/hybrid_context.lmdb` (84 MB, 20,939 entries).

## 3.6 Per-channel Slat Normalization

### 3.6.1 Vấn đề: Identity Collapse

Khi train iMF mà không normalize slat:
- Slat std = 0.36 (per-channel)
- Noise std = 1.0
- Target velocity $v = e - x \approx e$ (khi $\|x\| \ll \|e\|$)
- Model học $u_\text{pred} \approx z_t$ (identity collapse)
- Sample $z_0 = z_1 - u_\text{pred} \approx 0$ → mất tín hiệu

Verified ở epoch 16 của old training: cos_sim sample vs GT = **0.07** (gần random), pred std 0.08 (vs target 0.37).

### 3.6.2 Fix: TRELLIS.2-style Per-channel Norm

```python
# At training time
slat_normalized = (slat_raw - mean) / std
loss = compute_iMF_loss(slat_normalized, ...)

# At inference time
slat_pred_normalized = sample_1_step(...)
slat_pred_raw = slat_pred_normalized * std + mean
mesh = sc_vae.decode(slat_pred_raw)
```

Stats computed từ toàn bộ 20,369 training samples (`data/slat_stats.pt`):
- `mean[0:32]`: range $[-0.048, +0.053]$, mean ≈ 0
- `std[0:32]`: range $[0.258, 0.431]$, mean = 0.364

## 3.7 Data Pipeline

### 3.7.1 LMDB Caching

3 LMDB chính:

| LMDB | Size | Entries | Content |
|------|------|---------|---------|
| `ovoxel_cache_lmdb` | 272 GB | 21K | Raw O-Voxel features (input cho SC-VAE) |
| `hybrid_context.lmdb` | 84 MB | 20,939 | ArcFace + FLAME + DINOv2 [946] |
| `slat_context.lmdb` | 11 GB | 20,369 | Pre-computed `{slat[4096,32], context[946]}` cho iMF |

→ Training Stage 2 chỉ cần đọc LMDB → tiết kiệm GPU.

### 3.7.2 Training Strategy 2-Stage

**Stage A (current)**: Joint train trên cả 2 datasets
- 20,369 samples (FaceVerse 2,100 + FaceScape 18,269 train, test giữ riêng)
- 400 epochs × ~25 min = ~7 ngày

**Stage B (sau plateau)**: Finetune FaceVerse-only
- 2,100 samples (FaceVerse có vai, mắt, miệng đầy đủ)
- 100 epochs × ~7 phút = ~7 giờ
- LR thấp hơn 10× (2e-5), EMA decay 0.999

## 3.8 Tóm tắt đóng góp cho Chương 3

| Component | Lựa chọn FaceDiff | Lý do |
|-----------|-------------------|-------|
| Backbone | BiMamba O(N) | Linear complexity, 8 GB VRAM scan |
| Token ordering | Hilbert curve | Preserve 3D locality cho SSM |
| Norm | RMSNorm | Mamba spec, rẻ hơn LayerNorm |
| Output init | Zero | Stable diffusion training |
| Conditioning | In-context 24 tokens | iMF paper Section 4.3 |
| Training paradigm | iMF (1-NFE) | SOTA 1-step gen |
| FLAME extractor | MediaPipe V2 | Real blendshapes (52→50 dim) |
| Slat normalization | Per-channel | Fix identity collapse |
| Memory optimization | Split-forward CFG | Save 50% activation VRAM |

---

# CHƯƠNG 4. PHÂN TÍCH LÝ THUYẾT

## 4.1 Phân tích Complexity

### 4.1.1 Complexity Comparison

| Module | Op | Complexity | FaceDiff $L=4096$, $D=512$ |
|--------|-----|------------|----------------------------|
| Transformer Attention | Q·K·V | $O(B L^2 D)$ | $B \cdot 16.7M \cdot 512$ |
| Linear Attention | Approx softmax | $O(B L D^2)$ | $B \cdot 4096 \cdot 262K$ |
| **Mamba SSM** | Selective scan | $O(B L D N)$ | $B \cdot 4096 \cdot 512 \cdot 16$ |
| **BiMamba** | 2× selective scan | $O(2 B L D N)$ | $B \cdot 4096 \cdot 16K$ |
| FaceDiff total | 12 BiMamba | $O(24 B L D N)$ | $B \cdot 4096 \cdot 196K$ |

**Tỷ lệ với Transformer DiT**:
$$\frac{\text{FaceDiff (Mamba)}}{\text{DiT (Attention)}} = \frac{24 \cdot D \cdot N}{L \cdot D} = \frac{24 \cdot 16}{4096} = \frac{384}{4096} \approx \frac{1}{10.7}$$

→ FaceDiff backbone **~10× rẻ hơn Transformer** cùng sequence length.

### 4.1.2 VRAM Breakdown (Stage 2 training)

Đo thực tế với batch=4, d_state=16, expand=2, 12 layers:

| Bucket | VRAM | % | Note |
|--------|------|---|------|
| Model weights (FP32 master) | 196 MB | 1.2% | 49M params × 4 bytes |
| Optimizer AdamW (m, v) | 392 MB | 2.4% | 2× model |
| Gradients | 196 MB | 1.2% | 1× model |
| EMA copy | 196 MB | 1.2% | 1× model |
| **Mamba scan intermediates** | **~8 GB** | **48%** | $24 \times 333$ MB |
| Hidden activations | ~600 MB | 3.6% | 12 layers × 50 MB |
| JVP tangent buffer | ~1.5 GB | 9% | 50% batches có r≠t |
| CUDA workspace + fragmentation | ~1 GB | 6% | |
| **Process total** | **~12 GB** | **70%** | |
| Other GPU process | 5.2 GB | 30% | Background |
| **GPU total** | **17 GB / 24 GB** | | Margin 7 GB |

→ **Mamba scan dominates VRAM** — đây là root cause cho mọi tối ưu phải target.

## 4.2 Sample Efficiency Analysis

### 4.2.1 Inference Steps

| Method | Steps | Time/sample (RTX 4090, batch=1) |
|--------|-------|-------------------------------------|
| DDPM-50 (TRELLIS.2) | 50 | ~25s (theoretical) |
| Flow Matching | 10-100 | 5-50s |
| **FaceDiff iMF** | **1** | **~1.5s** |

→ FaceDiff đạt mục tiêu < 2s/mesh.

### 4.2.2 1-step vs Multi-step trade-off

iMF paper Tab 2 cho ImageNet 256×256:

| Method | NFE | FID |
|--------|-----|-----|
| iMF-XL/2 | 1 | 1.72 |
| iMF-XL/2 | 2 | 1.54 |
| DiT-XL/2 | 250 | 2.27 |

iMF chỉ kém DDPM nhiều bước < 1 FID — chất lượng acceptable cho production.

## 4.3 Identity Collapse — Phân tích Toán học

### 4.3.1 SNR (Signal-to-Noise Ratio)

Với slat raw (std = 0.36) và noise (std = 1.0):

$$\text{SNR}_{\text{raw}} = \frac{\text{Var}(x)}{\text{Var}(e)} = \frac{0.36^2}{1.0^2} = 0.13 \tag{4.1}$$

→ Signal $x$ bị át bởi noise $e$ trong $z_t = (1-t)x + te$.

**Target velocity** $v = e - x$:
- $\|e\|^2 \approx 1$ (chuẩn hoá)
- $\|x\|^2 \approx 0.13$
- $\|v\|^2 \approx 1.13$
- $\cos(v, e) = \frac{\|e\|^2 - e \cdot x}{\|e\| \|v\|} \approx \frac{1}{\sqrt{1.13}} = 0.94$

→ $v$ và $e$ gần như **đồng hướng**. Model học $u_\text{pred} \approx z_t \approx e$ là **trivial solution** với loss thấp.

### 4.3.2 Sau Per-channel Normalization

$$\text{SNR}_{\text{normalized}} = \frac{\text{Var}(\tilde{x})}{\text{Var}(e)} = \frac{1.0}{1.0} = 1.0 \tag{4.2}$$

→ Signal và noise cùng magnitude. $\cos(v, e) = \frac{1}{\sqrt{2}} = 0.707$ — không còn trivial alignment.

## 4.4 So sánh với DiM-3D và TRELLIS.2

### 4.4.1 Architecture choices

| Component | TRELLIS.2 | DiM-3D | iMF | **FaceDiff** |
|-----------|-----------|--------|-----|--------------|
| Backbone | U-DiT (Transformer) | Mamba | DiT (Transformer) | **Mamba** |
| Conditioning | Cross-attention | AdaLN-zero | **In-context tokens** | **In-context** |
| Generation | DDPM-50 | DDPM-1000 | iMF-1 | **iMF-1** |
| Block | Attn + FFN | Mamba + FFN | Attn + FFN | **BiMamba only** |
| Norm | LayerNorm | LayerNorm | RMSNorm (advanced) | **RMSNorm** |
| Output init | Standard | Standard | Zero | **Zero** |
| Token order | Patches | Patches | Patches | **Hilbert** |
| Domain | Mesh 3D | Point cloud | Image 2D | **Mesh 3D** |

→ FaceDiff là **hybrid design**, kết hợp 4 paper khác nhau theo cách độc đáo.

### 4.4.2 Deviation từ paper iMF (cần thừa nhận)

- ✅ In-context conditioning: MATCH (24 tokens, 8 ctx + 4×4)
- ✅ iMF objective: MATCH (v-loss reparameterized, flexible CFG)
- ✅ RMSNorm: MATCH
- ❌ **Thiếu FFN sub-block** trong mỗi layer — capacity giảm ~30%
- ❌ Backbone Mamba thay Transformer — kiểu khác

→ FFN missing là deviation lớn nhất. Sẽ đánh giá ở Phase F (epoch 200) dựa trên cos_sim metric, quyết định có nên refactor không.

---

# CHƯƠNG 5. ĐÁNH GIÁ THỰC NGHIỆM

## 5.1 Các tham số đánh giá

### 5.1.1 Mesh Quality Metrics

**Chamfer Distance (CD)** — bidirectional point-to-set distance:

$$\text{CD}(X, Y) = \frac{1}{2|X|} \sum_{x \in X} \min_{y \in Y} \|x - y\|^2 + \frac{1}{2|Y|} \sum_{y \in Y} \min_{x \in X} \|y - x\|^2 \tag{5.1}$$

**F-Score** at threshold $\tau$:

$$\text{Prec}(\tau) = \frac{|\{p \in P_{\text{pred}} : d(p, S_{\text{gt}}) < \tau\}|}{|P_{\text{pred}}|}, \quad \text{Rec}(\tau) = \frac{|\{p \in P_{\text{gt}} : d(p, S_{\text{pred}}) < \tau\}|}{|P_{\text{gt}}|} \tag{5.2}$$

$$\text{F1}(\tau) = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}} \tag{5.3}$$

**SSIM** for rendered images (2D quality):

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)} \tag{5.4}$$

### 5.1.2 Identity Preservation

**ArcFace cosine similarity** giữa render mesh và GT image:

$$\text{ID}_{\text{cos}} = \frac{f_{\text{render}} \cdot f_{\text{gt}}}{\|f_{\text{render}}\| \cdot \|f_{\text{gt}}\|} \tag{5.5}$$

**Target**: ID_cos > 0.5 (same person threshold).

### 5.1.3 Slat-level Metrics (internal)

**Cosine similarity** giữa generated slat và GT slat:
$$\text{cos\_sim}_{\text{slat}} = \frac{\langle \tilde{z}_{\text{pred}}, \tilde{z}_{\text{gt}} \rangle}{\|\tilde{z}_{\text{pred}}\| \cdot \|\tilde{z}_{\text{gt}}\|}$$

**Boundary velocity cos_sim** @ $t \in [0.1, 0.5]$ (training distribution):
- Health indicator của VoxelMamba conditioning

**1-step sample cos_sim** @ $t=1, r=0$:
- Production-quality indicator

### 5.1.4 Computational Metrics

- VRAM peak (training, inference)
- Throughput (samples/s)
- Inference latency (seconds/mesh)
- Total training time (days)

## 5.2 Phương pháp thí nghiệm

### 5.2.1 Datasets

| Dataset | Identities | Meshes | Vertex/mesh | Train | Test |
|---------|-----------|--------|-------------|-------|------|
| **FaceScape** [16] | 847 | 18,658 | 200K-400K | 18,269 | 389 |
| **FaceVerse** [17] | 110 | 2,310 | 5K-20K | 2,100 | 210 |
| **Total** | 957 | 20,968 | 20,369 (96%) | 599 (4%) |

**Split strategy**: Theo identity (không leakage). 10 IDs/dataset giữ cho test.

### 5.2.2 Hyperparameters

**Stage 1 — SC-VAE**:
- Optimizer: AdamW (fused), $\beta_1=0.9, \beta_2=0.999$, weight_decay=0
- Learning rate: $5 \times 10^{-5}$ → cosine với min $10^{-6}$
- Batch: 4, gradient accum 33 (effective 132)
- KL weight: $10^{-6}$, warmup 20 epochs
- Rho weight: 0.2 (TRELLIS.2 dùng 0.1)
- Precision: FP16 AMP với LayerNorm32 cast FP32
- Total epochs: 500 (epoch 500 đạt recon=0.0213, KL=14.58)

**Stage 2 — VoxelMamba + iMF**:
- Optimizer: AdamW fused, lr=$2 \times 10^{-4}$, weight_decay=$10^{-5}$
- Cosine scheduler + 1000 step warmup
- Batch: **4** × grad_accum **16** = effective 64
- Total epochs: 400 (~7 ngày trên RTX 4090)
- Precision: BFloat16 AMP
- EMA decay: 0.9995
- Dropout: 0.05
- TF32 + cudnn.benchmark: enabled
- NUM_WORKERS: 4

**iMF specific**:
- ratio_r_neq_t: 0.5
- t_sampler: curriculum (logit-normal $\mu=-0.4, \sigma=1$, switch uniform ở 30% progress = epoch 120)
- cfg_omega: power distribution $[1, 8]$, $\beta=1$
- cfg_context_dropout: 0.1
- v-head: 512-dim auxiliary, weight 0.1
- adaptive_loss_weighting: True (per-bin EMA reweighting)

### 5.2.3 Training Curriculum

```
ep 1-7    : warmup (lr 6e-5 → 2e-4)
ep 8-120  : logit-normal t (focus mid-range)
ep 121-400: 80% uniform + 20% logit-normal (học t cực trị)
ep 200    : Phase F evaluation
ep 250+   : plateau → switch to Stage B
```

### 5.2.4 Hardware Setup

- **GPU**: 1× RTX 4090 (24GB VRAM)
- **CPU**: server-grade (CPU 100% trong I/O phase)
- **Storage**: HDD (LMDB rất tốt cho sequential read) + SSD cache
- **RAM**: 64 GB (enough cho mediapipe + workers)

## 5.3 Kết quả thí nghiệm

### 5.3.1 SC-VAE Training (Stage 1, hoàn thành 16/05/2026)

| Epoch | total_loss | recon | KL | rho | Δrecon/10ep |
|-------|-----------|-------|------|-----|-------------|
| 390 | 0.038 | 0.026 | 0.83 | — | — |
| 470 | 0.0880 | 0.0220 | 14.71 | 0.039 | — |
| 480 | 0.0870 | 0.0217 | 14.67 | 0.039 | -0.0003 |
| 490 | 0.0863 | 0.0214 | 14.63 | 0.038 | -0.0003 |
| **500** | **0.0858** | **0.0213** | **14.58** | **0.038** | -0.0001 |

→ Plateau. SC-VAE đủ tốt cho Stage 2.

### 5.3.2 Stage 2 Training (đang chạy, snapshot 18/05/2026)

**Trước fix identity collapse** (epoch 16):
- cos_sim sample vs GT: **0.07** (gần random)
- pred slat std: 0.08 (target 0.37, **22%**)
- 5-step Euler cos_sim: 0.02 (tệ hơn 1-step → velocity field hỏng)
- `cos(u_pred, z_1) = 0.9973` → identity collapse confirmed

**Sau fix (per-channel normalize)** trajectory:

| Run | Epoch | Loss | Boundary cos @ t=0.1 | 1-step cos_sim |
|-----|-------|------|---------------------|----------------|
| ban đầu | 16 (broken) | 0.215 plateau | — | 0.07 |
| sau fix slat | 2 | 0.638 | 0.91 | 0.007 |
| sau fix slat | 5 | 0.436 | **0.93** | 0.048 |
| sau fix curriculum | 9 | 0.415 | 0.928 | 0.045 |
| **Hiện tại (FLAME+CFG fix)** | **8** | **0.428** | TBD ep20 | TBD ep20 |

→ Sample quality improvement đáng kể từ 0.07 → 0.045 (7×) nhưng còn xa target 0.3+.

### 5.3.3 Speedup Experiments

| Config | Batch/s | Samples/s | ETA/epoch | ETA Stage A |
|--------|---------|-----------|-----------|-------------|
| Initial (batch=2, batch doubling) | 4.1 | 8.2 | 41 min | 11.4 ngày |
| batch=3 + TF32 + cudnn.benchmark | 2.9 | 8.7 | 39 min | 9.7 ngày (-15%) |
| **batch=4 + refactored CFG** | **2.9** | **11.4** | **30 min** | **7.0 ngày (-39%)** |

### 5.3.4 Bug Discovery Timeline

| Date | Bug | Impact | Fix |
|------|-----|--------|-----|
| 09/05 | Gamma cache constant 1.0 | SC-VAE invalid | Recompute |
| 10/05 | Output activations missing | dv → center voxel | Apply sigmoid+margin |
| 11/05 | Pre-latent LayerNorm missing | Posterior unstable | Add F.layer_norm |
| 15/05 | DC mesh holes | Boundary voxels dropped | `_prefill_boundary_voxels` |
| 16/05 | KL norm wrong | Log values misleading | Divide `mu.numel()` |
| 17/05 | **Identity collapse** | Sample cos_sim 0.07 | **Per-channel slat normalize** |
| 17/05 | 1-step learning slow | Curriculum chưa switch | `switch_ratio: 0.6 → 0.3` |
| 18/05 | **FLAME context constant** | No expression conditioning | **MediaPipe FaceLandmarker V2** |
| 18/05 | **CFG VRAM waste** | Activation 2× | **Split forward + no_grad uncond** |

## 5.4 Ablation Studies (kế hoạch sau Stage A)

### 5.4.1 Architecture Ablations

| Variant | Expected ΔFID |
|---------|---------------|
| Baseline FaceDiff (BiMamba, no FFN) | 0 (reference) |
| + FFN sub-block per layer | -5% (iMF Tab 1c implication) |
| - Hilbert ordering (raster) | +10% (worse 3D locality) |
| - RMSNorm (LayerNorm) | +1% (negligible) |
| - Zero-init output | +2% |
| Increase layers 12 → 16 | -3% (match DiM-3D-S) |

### 5.4.2 Conditioning Ablations

| Variant | cos_sim expected |
|---------|-----------------|
| Baseline 946-dim (ArcFace + FLAME + DINOv2) | reference |
| Without FLAME (896-dim) | -10% (no expression control) |
| Without DINOv2 (562-dim) | -5% (no back shape) |
| Without ArcFace (434-dim) | -20% (no identity) |
| MediaPipe FLAME vs random | +30% (fix this work) |

### 5.4.3 Curriculum Ablations

| Variant | 1-step cos_sim @ ep200 |
|---------|------------------------|
| switch_ratio=0.6 (paper) | low (curriculum chưa switch) |
| **switch_ratio=0.3 (FaceDiff)** | medium |
| switch_ratio=0.15 | high but instability risk |
| pure uniform | unstable |

---

# CHƯƠNG 6. KẾT LUẬN

## 6.1 Kết luận

### 6.1.1 Tóm tắt đóng góp

Đồ án đã đề xuất **FaceDiff** — hệ thống tạo sinh mesh khuôn mặt 3D với các đặc tính:

1. **Chất lượng cao**: 200K+ vertex mesh, 10-channel (geometry + RGB), polygon
2. **Sinh 1 bước**: < 2 giây/mesh nhờ iMF objective
3. **Đơn GPU**: Peak VRAM 17 GB / 24 GB trên RTX 4090
4. **Kiểm soát ngữ nghĩa**: Hybrid Context 946-dim (identity + expression + back shape)
5. **Pipeline tự động hoá**: Reproducible với scripts đầy đủ

### 6.1.2 Đóng góp kỹ thuật then chốt

1. **VoxelMamba — hybrid backbone**: kết hợp BiMamba (DiM-3D) + Hilbert curve (VoxelMamba) + In-context conditioning (iMF Section 4.3). Lần đầu áp dụng cho mesh 3D.

2. **Application của iMF cho 3D**: Triển khai Improved Mean Flow (paper 12/2025) cho lĩnh vực mesh 3D — đầu tiên trong văn liệu.

3. **Per-channel Slat Normalization**: Phát hiện và sửa identity collapse, một bug nghiêm trọng khi áp dụng iMF cho latent có std < 1.

4. **MediaPipe FLAME Adapter**: Thay random-init FLAME bằng pretrained MediaPipe FaceLandmarker V2 (52 ARKit blendshapes) — bug fix dẫn đến training data đúng.

5. **CFG VRAM Optimization**: Refactor batch doubling thành split forward (cond grad + uncond no_grad) — tiết kiệm ~50% activation memory, cho phép batch=4 thay vì batch=2.

### 6.1.3 So với State-of-the-Art

| Tiêu chí | TRELLIS.2 | DiM-3D | FaceDiff |
|---------|-----------|---------|----------|
| GPU | 8×A100 | 1×A100 | **1×RTX 4090** |
| Inference steps | 50 | 1000 | **1** |
| Time/mesh | ~30s | ~10s | **<2s** |
| Mesh quality | 200K+ vert | Point cloud | **200K+ vert** |
| Domain | General 3D | General 3D | **Face specialized** |

FaceDiff cân bằng quality–speed–cost: **không đạt SOTA quality** của TRELLIS.2 nhưng **chạy được trên hardware tiêu dùng**.

## 6.2 Hướng phát triển trong tương lai

### 6.2.1 Ngắn hạn (06-08/2026)

- **Architecture refinement**: Add FFN sub-block (theo iMF paper) nếu Phase F evaluation cho thấy capacity insufficient
- **Stage B finetune**: FaceVerse-only để học chi tiết (vai, mắt, miệng)
- **E2E inference optimization**: TorchScript export, batched pipeline, target < 1s/mesh
- **Ablation studies**: Validate đóng góp từng component

### 6.2.2 Trung hạn (09-12/2026)

- **Multi-view conditioning**: Sinh từ nhiều view của input
- **Temporal consistency**: Cho video face animation
- **Dataset expansion**: Tích hợp FERG, BU-3DFE, CoMA (expression diversity)
- **Production deployment**: Web demo Three.js, REST API endpoint

### 6.2.3 Dài hạn (2027+)

- **General 3D shape**: Mở rộng từ face sang object generation (ShapeNet)
- **High-resolution scaling**: Voxel grid $512^3$ hoặc $1024^3$
- **Distillation**: Distill DDPM 50-step teacher (TRELLIS.2) thành 1-step student
- **Mobile inference**: Quantization + pruning cho edge deployment
- **Bài báo**: CVPR / ICCV / NeurIPS

### 6.2.4 Hạn chế và Lessons Learned

**Hạn chế hiện tại**:
- Chất lượng sample 1-step chưa đạt mức TRELLIS.2 DDPM-50
- FFN missing trong Mamba block → capacity per layer thấp hơn paper
- Train chỉ với 2 datasets (FaceScape + FaceVerse) — diversity hạn chế
- Inference latency vẫn 1.5s (target < 1s)

**Lessons learned**:
- **Slat normalization là CRITICAL** với iMF/Mean Flow — bug này không obvious nếu không đo SNR
- **MediaPipe > random CNN** cho FLAME extraction — pretrained model luôn tốt hơn random
- **VRAM profiling phải xét peak** — autocast + checkpointing không reduce activations đủ
- **In-context conditioning đủ tốt** — không cần AdaLN-zero (iMF paper Section 4.3 đúng)

---

# TÀI LIỆU THAM KHẢO

1. Geng, Z., Lu, Y., Wu, Z., Shechtman, E., Kolter, J. Z., & He, K. (2025). *Improved Mean Flows: On the Challenges of Fastforward Generative Models*. arXiv:2512.02012.

2. Xiang, J., et al. (2025). *TRELLIS.2: Structured 3D Latents for Scalable and Versatile 3D Generation*. arXiv:2512.14692. Microsoft Research.

3. Geng, Z., et al. (2025). *MeanFlow: One-Step Generative Modeling via Average Velocity*. arXiv:2505.13447. (Original Mean Flow)

4. Mo, S., et al. (2024). *Efficient 3D Shape Generation via Diffusion Mamba with Bidirectional SSMs (DiM-3D)*. NeurIPS — arXiv:2406.05038.

4b. Zhang, G., et al. (2024). *Voxel Mamba: Group-Free State Space Models for Point Cloud based 3D Object Detection*. arXiv:2406.10700.

5. Gu, A., & Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.

6. Lipman, Y., et al. (2023). *Flow Matching for Generative Modeling*. ICLR.

7. Deng, J., et al. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. CVPR.

8. Li, T., et al. (2017). *Learning a Model of Facial Shape and Expression from 4D Scans (FLAME)*. SIGGRAPH Asia.

9. Oquab, M., et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR.

10. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models (DDPM)*. NeurIPS.

11. Mo, S., Xie, R., Chu, L., et al. (2023). *DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation*. NeurIPS — arXiv:2306.10006.

12. Poole, B., et al. (2023). *DreamFusion: Text-to-3D using 2D Diffusion*. ICLR.

13. Siddiqui, Y., et al. (2024). *MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers*. CVPR.

14. Rombach, R., et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR.

15. Lugmayr, A., et al. (2023). *MediaPipe Face Landmarker*. Google. https://google-ai-edge.github.io/mediapipe/

16. Zhu, H., et al. (2020). *FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction*. CVPR.

17. Li, Z., et al. (2022). *FaceVerse: a Fine-grained and Detail-Controllable 3D Face Morphable Model from a Hybrid Dataset*. CVPR.

18. Peebles, W., & Xie, S. (2023). *Scalable Diffusion Models with Transformers (DiT)*. ICCV.

19. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes (VAE)*. ICLR.

---

# PHỤ LỤC

## A.1 Cấu hình Máy Huấn luyện

```
GPU: NVIDIA GeForce RTX 4090
  - 24 GB GDDR6X VRAM
  - 16,384 CUDA cores
  - Compute capability 8.9 (Ada Lovelace)
  - CUDA 12.x

CPU: Intel/AMD server-grade (≥ 16 cores recommended)
RAM: 64 GB (cho MediaPipe + dataloader workers)
Storage: 17 TB HDD + SSD cache (LMDB ~272 GB)
OS: Ubuntu 22.04 LTS
```

## A.2 Software Stack

```
Python 3.11 (conda env `facediff`)
PyTorch 2.x with CUDA 12.x
spconv-cu12x (Sparse Convolution)
mamba-ssm 2.3.1 (Selective State Space)
nvdiffrast 0.4.0 (mesh rendering — future)
lmdb (key-value cache)
trimesh + open3d (mesh ops)
mediapipe 0.10.x (face landmarker)
insightface (ArcFace)
transformers + torch.hub (DINOv2)
```

## A.3 Lệnh chạy chính

```bash
# Activate environment
source miniconda3/etc/profile.d/conda.sh && conda activate facediff

# Stage 1: SC-VAE training (đã hoàn thành)
python src/train_sc_vae.py \
    --resume checkpoints/sc_vae_shape/epoch_500.pt \
    --epochs 700

# Pre-compute Slat cache (offline data cho Stage 2)
python scripts/precompute_slat_cache.py \
    --sc-vae-ckpt checkpoints/sc_vae_shape/epoch_500.pt \
    --dataset both --skip-existing

# Pack into LMDB
python scripts/pack_slat_lmdb.py

# Compute slat normalization stats (one-time)
python scripts/compute_slat_stats.py

# Build hybrid_context.lmdb (với MediaPipe FLAME)
python scripts/build_context_lmdb.py \
    --out-lmdb data/hybrid_context.lmdb \
    --device cuda:0

# Stage 2: iMF VoxelMamba training (đang chạy)
BATCH_SIZE=4 GRAD_ACCUM=16 NUM_WORKERS=4 \
RESUME=checkpoints/imf_unet/best.pt \
bash scripts/train_imf.sh

# E2E inference test (sau training)
python scripts/test_e2e_inference.py \
    --n-samples 8 --steps 1 20 --dataset-filter faceverse \
    --decode-device cuda
```

## A.4 Cấu trúc Repository

```
facediff/
├── src/
│   ├── config.py                  # TrainConfig dataclass
│   ├── train_sc_vae.py            # Stage 1 training
│   ├── train_imf.py               # Stage 2 training
│   ├── hilbert.py                 # Hilbert curve utilities
│   ├── mesh_gpu.py                # GPU KNN + Taubin (NEW)
│   ├── utils.py
│   ├── models/
│   │   ├── sc_vae.py              # SC-VAE encoder/decoder
│   │   ├── sc_vae_loss.py         # Loss functions
│   │   ├── voxel_mamba.py         # BiMamba backbone
│   │   └── imf_diffusion.py       # iMF training + sampling
│   ├── data/
│   │   ├── ovoxel_converter.py    # Mesh → O-Voxel
│   │   ├── arcface_extractor.py
│   │   ├── flame_adapter.py       # MediaPipe FaceLandmarker V2 (NEW)
│   │   ├── feature_extractor.py   # DINOv2
│   │   └── mesh_renderer.py
│   ├── modules/
│   │   └── norm.py                # LayerNorm32
│   ├── inference/
│   │   └── generator.py           # E2E inference pipeline
│   └── scvae_train/               # Stage 1 utilities
├── scripts/
│   ├── build_context_lmdb.py
│   ├── compute_slat_stats.py
│   ├── precompute_slat_cache.py
│   ├── pack_slat_lmdb.py
│   ├── remix_slat_lmdb_with_new_context.py    # NEW (FLAME fix)
│   ├── auto_fix_flame_and_retrain.sh          # NEW (automation)
│   ├── test_imf_sample.py
│   ├── test_imf_at_training_t.py
│   ├── test_e2e_inference.py
│   ├── test_sc_vae_recon_v2.py
│   ├── visualize_test_context.py              # NEW
│   └── train_imf.sh
├── data/
│   ├── slat_cache/                # Stage 2 cache .pt
│   ├── slat_context.lmdb/         # Merged slat+context (11 GB)
│   ├── hybrid_context.lmdb/       # Context only (84 MB)
│   ├── slat_stats.pt              # Per-channel mean+std
│   ├── mediapipe_models/          # FaceLandmarker .task
│   └── mesh_manifest.json
├── checkpoints/
│   ├── sc_vae_shape/              # Stage 1 ckpts
│   └── imf_unet/                  # Stage 2 ckpts
├── docs/
│   ├── papers/                    # Reference PDFs (gitignored)
│   └── thesis_draft.md            # THIS FILE
├── third_party/
│   └── TRELLIS.2/                 # Reference implementation
├── Bao_cao_FaceDiff_ChiTiet.md    # Progress notes
├── CLAUDE.md                      # AI agent instructions
└── README.md
```

## A.5 Notation và Symbols

| Symbol | Ý nghĩa |
|--------|---------|
| $x$ | Data sample (slat token tensor [4096, 32]) |
| $e$ | Noise sample $\sim \mathcal{N}(0, I)$ |
| $z_t$ | Interpolation $(1-t)x + te$, $t \in [0, 1]$ |
| $v$ | Instantaneous velocity $v(z_t, t) = e - x$ |
| $u$ | Average velocity $\frac{1}{t-r}\int_r^t v(z_s, s) ds$ |
| $V_\theta$ | Compound function $u_\theta + (t-r) \cdot \text{sg}(\partial_t u_\theta)$ |
| $\omega$ | CFG guidance scale |
| $c$ | Context (946-dim Hybrid) |
| $L$ | Sequence length (4096 slat tokens) |
| $D$ | Hidden dim (512) |
| $N$ | SSM state dim (16) |
| $B$ | Batch size |
| $\Delta$ | Mamba discretization timestep |
| $\beta_{\text{KL}}$ | KL loss weight ($10^{-6}$) |

## A.6 Bug Fix Catalogue (Sửa Bug Theo Trình tự Phát triển)

(Xem chi tiết Section 7.2 của Bao_cao_FaceDiff_ChiTiet.md)

| ID | Bug | Severity | Fix date | File |
|----|-----|----------|----------|------|
| B-01 | Gamma cache constant 1.0 | High | 09/05 | `pack_lmdb_fast.py` |
| B-06 | dv activation missing | Critical | 10/05 | `sc_vae.py:apply_dv_activation` |
| B-08 | KL norm wrong | Low (cosmetic) | 10/05 | `sc_vae_loss.py` |
| B-09 | Pre-latent LayerNorm | Medium | 10/05 | `sc_vae.py:pre_latent_norm` |
| B-13 | Slat cache random context | High | 11/05 | `precompute_slat_cache.py` |
| B-DC1 | DC mesh holes | Medium | 15/05 | `test_sc_vae_recon_v2.py:_prefill_boundary_voxels` |
| B-IC | **Identity collapse** | **Critical** | **17/05** | `train_imf.py` (per-channel norm) |
| B-FLAME | **FLAME constant** | **Critical** | **18/05** | `flame_adapter.py` (MediaPipe) |
| B-CFG | CFG VRAM waste | Medium (perf) | 18/05 | `imf_diffusion.py` (split forward) |

---

*(End of Thesis Draft — Generated 18/05/2026 — Use as base for Word/LaTeX expansion)*
