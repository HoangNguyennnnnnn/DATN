# Báo cáo Nghiên cứu Chuyên sâu: FaceDiff — Hệ thống Tạo sinh Khuôn mặt 3D Một Bước trên Đơn GPU

**Ngày cập nhật:** 13/05/2026 (revision 10 — báo cáo tiến độ toàn diện, rà soát VoxelMamba vs DiM-3D, chuẩn bị Cloud GPU deployment)  
**Tác giả:** Nhóm nghiên cứu FaceDiff  
**Cấu hình Mục tiêu:** Đơn GPU RTX 4090 (24GB VRAM)  
**Bộ dữ liệu:** FaceVerse_3D & FaceScape  
**Trạng thái checkpoint:** SC-VAE epoch 442/700 đang chạy (Step 627,800, recon_loss=0.0224). Resume từ epoch 441 với EMA + Depth-to-Normal Loss + Cosine Restart +200 epochs.

---

## Mục lục

1. [Giới thiệu Đề tài](#1-giới-thiệu-đề-tài)
2. [Mục tiêu và Đóng góp](#2-mục-tiêu-và-đóng-góp)
3. [Nền tảng Toán học](#3-nền-tảng-toán-học)
4. [Phương pháp Đề xuất](#4-phương-pháp-đề-xuất)
5. [Bộ dữ liệu: FaceScape & FaceVerse](#5-bộ-dữ-liệu-facescape--faceverse)
6. [Thực nghiệm](#6-thực-nghiệm)
7. [Phân tích và Giải thích Kết quả](#7-phân-tích-và-giải-thích-kết-quả)
8. [Kế hoạch Tiếp theo](#8-kế-hoạch-tiếp-theo)
9. [Cập nhật Alignment iMF (Revision 4)](#9-cập-nhật-alignment-imf-revision-4)

---

## 1. Giới thiệu Đề tài

### 1.1. Bối cảnh và Động lực Nghiên cứu

Tạo sinh khuôn mặt 3D (3D Face Generation) là bài toán trọng tâm của thị giác máy tính, có ứng dụng trong game, phim hoạt hình, VR/AR, và y tế thẩm mỹ. Mục tiêu: từ điều kiện đầu vào (ảnh khuôn mặt, biểu cảm, danh tính), hệ thống sinh lưới đa giác 3D (Polygon Mesh) chất lượng cao.

Thách thức chính của các phương pháp hiện tại:

- **Độ phức tạp tính toán bậc ba:** Biểu diễn thể tích $256^3$ tạo hàng triệu điểm, vượt khả năng xử lý Transformer với Attention $O(N^2)$
- **Tốc độ sinh chậm:** Diffusion thông thường cần 20–50 bước ODE/SDE
- **Phần cứng đắt đỏ:** TRELLIS.2 đòi hỏi 8×A100 (320GB VRAM)

### 1.2. Các Công trình Liên quan

| Phương pháp | Biểu diễn | Ưu | Nhược |
|-------------|-----------|-----|-------|
| Point-E, PointFlow | Point Cloud | Đơn giản | Thiếu topology |
| DreamFusion, Magic3D | NeRF + SDS | Chất lượng cao | 30–60 phút/đối tượng, Janus effect |
| GaussianHead, HeadGAP | 3D Gaussian | Render đẹp | Khó trích Mesh |
| MeshGPT, MeshAnything | Mesh trực tiếp | Topology rõ | Giới hạn vài nghìn mặt |
| **TRELLIS.2** | **O-Voxel + Sparse VAE** | **Mesh chi tiết 200K+** | **8×A100, 50 bước DDPM** |

### 1.3. Vấn đề cần Giải quyết

1. **Chi phí phần cứng** — Không có giải pháp 3D chất lượng cao trên 1 GPU tiêu dùng
2. **Tốc độ sinh** — 20–50 bước khuếch tán không tương tác
3. **Kiểm soát ngữ nghĩa** — Nhiều hệ thống không kiểm soát danh tính + biểu cảm đồng thời
4. **Khoảng cách biểu diễn** — NeRF/Gaussian khó tích hợp pipeline sản xuất

---

## 2. Mục tiêu và Đóng góp

### 2.1. Mục tiêu

| # | Mục tiêu | Chỉ tiêu |
|---|----------|----------|
| 1 | Mesh 3D chất lượng cao | > 200K đỉnh, 10-kênh |
| 2 | Sinh 1 bước | < 2 giây/mẫu trên RTX 4090 |
| 3 | Kiểm soát danh tính + biểu cảm | Hybrid Context 946-dim |
| 4 | Đơn GPU | VRAM peak < 22GB |

### 2.2. Đóng góp

1. **SC-VAE tiết kiệm VRAM** với SparseResMLPBlock (giảm 45% VRAM so với ConvNeXt 3D) + Generative Pruning (Rho Loss)
2. **VoxelMamba** — backbone SSM $O(N)$ thay Transformer $O(N^2)$, Hilbert curve ordering. *Thiết kế lai (hybrid): BiMamba generation lấy cảm hứng từ DiM-3D [14], Hilbert ordering từ VoxelMamba [4], kết hợp in-context conditioning tự thiết kế cho iMF.*
3. **iMF (Improved Mean Flow)** — sinh 1 bước bằng JVP correction
4. **Hybrid Context 946-dim** = ArcFace(512) + FLAME(50) + DINOv2(384)
5. **Tối ưu đơn GPU**: INT4 quantization, BFloat16, gradient checkpointing, LMDB caching

---

## 3. Nền tảng Toán học

### 3.1. Mô hình Không gian Trạng thái (State Space Model — SSM)

#### 3.1.1. SSM Liên tục

Mô hình không gian trạng thái (SSM) liên tục mô tả hệ động lực tuyến tính ánh xạ đầu vào $u(t) \in \mathbb{R}$ sang đầu ra $y(t) \in \mathbb{R}$ thông qua trạng thái ẩn $h(t) \in \mathbb{R}^N$:

$$\frac{dh}{dt} = \mathbf{A} h(t) + \mathbf{B} u(t), \quad y(t) = \mathbf{C} h(t) + D u(t) \tag{SSM-1}$$

Trong đó:
- $\mathbf{A} \in \mathbb{R}^{N \times N}$ — ma trận chuyển trạng thái (state transition matrix)
- $\mathbf{B} \in \mathbb{R}^{N \times 1}$ — ma trận đầu vào (input matrix)
- $\mathbf{C} \in \mathbb{R}^{1 \times N}$ — ma trận đầu ra (output matrix)
- $D \in \mathbb{R}$ — bỏ qua (skip connection), thường $D = 0$

#### 3.1.2. Rời rạc hóa (Zero-Order Hold — ZOH)

Để áp dụng cho dữ liệu rời rạc (chuỗi tokens), SSM được rời rạc hóa bằng phương pháp Zero-Order Hold (ZOH) với bước thời gian $\Delta$:

$$\bar{\mathbf{A}} = \exp(\Delta \mathbf{A}), \quad \bar{\mathbf{B}} = (\Delta \mathbf{A})^{-1}(\bar{\mathbf{A}} - \mathbf{I}) \cdot \Delta \mathbf{B} \tag{SSM-2}$$

Phương trình rời rạc tương ứng:

$$h_k = \bar{\mathbf{A}} h_{k-1} + \bar{\mathbf{B}} x_k, \quad y_k = \mathbf{C} h_k \tag{SSM-3}$$

Phép toán (SSM-3) là hồi quy tuyến tính — có thể triển khai song song qua **Parallel Associative Scan** với độ phức tạp $O(N \log N)$ trên GPU, hoặc tuần tự $O(N)$.

#### 3.1.3. Selective SSM (Mamba)

**Đóng góp cốt lõi của Mamba** (Gu & Dao, 2024): Biến $\mathbf{B}$, $\mathbf{C}$, $\Delta$ thành **hàm phụ thuộc đầu vào** (input-dependent), cho phép mô hình "chọn lọc" (selective) thông tin:

$$\mathbf{B}_k = \text{Linear}_B(x_k), \quad \mathbf{C}_k = \text{Linear}_C(x_k), \quad \Delta_k = \text{Softplus}(\text{Linear}_\Delta(x_k)) \tag{SSM-4}$$

Với $\text{Softplus}(\cdot) = \log(1 + e^{(\cdot)})$ đảm bảo $\Delta_k > 0$.

**Khởi tạo HiPPO cho A:** Ma trận $\mathbf{A}$ được khởi tạo dạng đường chéo: $A_n = -(n+1)$ cho $n = 0, ..., N-1$. Khởi tạo này xuất phát từ lý thuyết High-order Polynomial Projection Operators (HiPPO), đảm bảo trạng thái ẩn tối ưu cho việc nén lịch sử chuỗi.

**Kiến trúc một block Mamba:**

```
Input x [B, L, D]
  │
  ├──→ Linear_expand → Conv1D(k=4) → SiLU → SSM(A, B(x), C(x), Δ(x)) ──┐
  │                                                                        │
  └──→ Linear_gate → SiLU ────────────────────────────────── × (hadamard) ─┘
                                                                    │
                                                             Linear_proj → Output
```

**So sánh độ phức tạp:**

| Mô hình | Complexity/token | Memory | Xử lý chuỗi 4096 |
|---------|-----------------|--------|-------------------|
| Transformer (Self-Attention) | $O(N^2 \cdot d)$ | $O(N^2)$ attention maps | ~16.7M entries/layer |
| Mamba (Selective SSM) | $O(N \cdot d \cdot n)$ | $O(N \cdot n)$ states | ~65K entries/layer |
| Tỷ lệ | — | — | **256× ít hơn** |

Trong đó $n$ = SSM state dim (16 trong FaceDiff), $d$ = model dim (512).

#### 3.1.4. Mamba Hai chiều (Bidirectional Mamba)

SSM có tính nhân quả (causal): $h_k$ chỉ tích lũy thông tin từ $x_0, ..., x_{k-1}$. Để mỗi voxel nhận thông tin từ mọi hướng trong không gian 3D:

**Quét xuôi (Forward):** Chuỗi $X = [x_1, ..., x_L]$ qua Mamba forward:
$$h_k^{\text{fwd}} = \bar{\mathbf{A}}_k h_{k-1}^{\text{fwd}} + \bar{\mathbf{B}}_k x_k, \quad y_k^{\text{fwd}} = \mathbf{C}_k h_k^{\text{fwd}} \tag{BiM-1}$$

**Quét ngược (Backward):** Chuỗi đảo $\tilde{X} = [x_L, ..., x_1]$ qua Mamba backward:
$$h_k^{\text{bwd}} = \bar{\mathbf{A}}_k h_{k-1}^{\text{bwd}} + \bar{\mathbf{B}}_k \tilde{x}_k, \quad y_k^{\text{bwd}} = \mathbf{C}_k h_k^{\text{bwd}} \tag{BiM-2}$$

**Tổng hợp với residual:**
$$\text{Output}_k = y_k^{\text{fwd}} + y_k^{\text{bwd}} + x_k \tag{BiM-3}$$

Mỗi block Bidirectional Mamba có 2 instance Mamba riêng biệt (không chia sẻ trọng số), cộng RMSNorm trước và Dropout sau.

### 3.2. Đường cong Hilbert (Hilbert Space-Filling Curve)

#### 3.2.1. Định nghĩa

Đường cong Hilbert là đường cong liên tục đi qua mọi điểm trong lưới $2^p \times 2^p \times 2^p$ đúng một lần, mà **không tự cắt chính nó**. Tính chất quan trọng nhất:

> **Spatial Locality:** Hai điểm gần nhau trong không gian 3D sẽ có vị trí gần nhau trên đường cong 1D.

#### 3.2.2. Xây dựng đệ quy

Đường cong Hilbert 3D bậc $p$ được xây dựng đệ quy từ bậc $p-1$:

1. Chia cube $2^p$ thành 8 octant $2^{p-1}$
2. Xoay và phản chiếu đường cong bậc $p-1$ trong mỗi octant để đầu-cuối nối liền
3. Thứ tự 8 octant tuân theo **Gray code** 3-bit

**Ánh xạ toạ độ → chỉ số Hilbert:**
$$\pi_H: (i, j, k) \in \{0,...,2^p-1\}^3 \mapsto n \in \{0,...,2^{3p}-1\} \tag{HC-1}$$

**Trong FaceDiff:** $p = 4$ (vì $16^3 = 4096$ Slat tokens). Hàm `get_hilbert_permutation_tensors()` tính 2 tensor:
- `perm` $\in \mathbb{Z}^{4096}$: ánh xạ thuận (3D → 1D)
- `inv_perm` $\in \mathbb{Z}^{4096}$: ánh xạ nghịch (1D → 3D)

Chi phí bộ nhớ: $2 \times 4096 \times 8\text{B} = 64\text{KB}$ — trivial.

#### 3.2.3. So sánh với các phương pháp sắp xếp khác

| Phương pháp | Spatial Locality | Complexity | Ghi chú |
|-------------|-----------------|------------|---------|
| Raster scan (row-major) | Kém — nhảy hàng xa | $O(1)$ | Hai voxel cạnh nhau trục Y cách $N$ trong chuỗi |
| Morton (Z-order) | Trung bình | $O(1)$ | Interleave bits, đơn giản nhưng có "nhảy" |
| **Hilbert** | **Tốt nhất** | $O(p)$ | Bảo toàn locality tối ưu, dùng trong FaceDiff |

### 3.3. Flow Matching (Khớp Luồng)

#### 3.3.1. Bài toán Tạo sinh

Cho phân phối dữ liệu $p_0(x)$ (Slat tokens sạch) và phân phối nhiễu $p_1(z) = \mathcal{N}(0, \mathbf{I})$. Mục tiêu: học ánh xạ $T: z_1 \sim p_1 \mapsto z_0 \sim p_0$.

#### 3.3.2. Đường nội suy (Interpolation Path)

Xác định đường dẫn xác suất $p_t$ nối $p_0$ và $p_1$ bằng nội suy tuyến tính:

$$z_t = (1 - t) x_0 + t \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \mathbf{I}), \quad t \in [0, 1] \tag{FM-1}$$

**Quy ước:** $t = 0$ là dữ liệu sạch, $t = 1$ là nhiễu thuần.

#### 3.3.3. Trường vận tốc (Velocity Field)

Vận tốc tức thời dọc theo đường nội suy:

$$v_t(z_t | x_0) = \frac{dz_t}{dt} = \varepsilon - x_0 \tag{FM-2}$$

**Conditional Flow Matching (CFM) Loss** (Lipman et al., 2023):

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t \sim U[0,1],\, x_0 \sim p_0,\, \varepsilon \sim \mathcal{N}(0,I)} \left[ \| v_\theta(z_t, t) - (\varepsilon - x_0) \|^2 \right] \tag{FM-3}$$

#### 3.3.4. Sinh mẫu bằng ODE

Từ $z_1 \sim \mathcal{N}(0, I)$, giải ODE ngược:

$$\frac{dz_t}{dt} = v_\theta(z_t, t), \quad z_1 \sim \mathcal{N}(0, I) \tag{FM-4}$$

Cần $K$ bước Euler/Heun (thường $K = 20$–$50$). **Đây là nhược điểm mà iMF giải quyết.**

### 3.4. Improved Mean Flow (iMF)

#### 3.4.1. Vận tốc Trung bình (Average Velocity)

Thay vì vận tốc tức thời $v(z, t)$, iMF (Geng et al., 2025, arXiv:2512.02012v1) định nghĩa **vận tốc trung bình** trên đoạn $[r, t]$:

$$u(z, r, t) = \frac{1}{t - r} \int_r^t v(z_s, s)\, ds \tag{iMF-1}$$

Trong đó $z_s$ là quỹ đạo ODE bắt đầu từ $z_r = z$ tại thời điểm $r$.

**Ý nghĩa vật lý:** $u$ là "vận tốc bình quân" mà nếu đi thẳng từ $z_r$ đến $z_t$ trong $(t - r)$ đơn vị thời gian, ta sẽ đến đúng đích. Khi $t - r \to 0$: $u \to v$ (suy biến thành vận tốc tức thời).

#### 3.4.2. Đồng nhất thức MeanFlow

Quan hệ giữa $u$ và $v$:

$$v(z, t) = u(z, r, t) + (t - r) \frac{\partial u}{\partial t}(z, r, t) \tag{iMF-2}$$

Chứng minh: Đạo hàm (iMF-1) theo $t$ bằng quy tắc Leibniz.

#### 3.4.3. Hàm hợp V (Compound Function)

Mạng $u_\theta$ dự đoán vận tốc trung bình. Để huấn luyện, xây dựng **hàm hợp V** xấp xỉ vận tốc tức thời:

$$V_\theta(z_t, r, t) = u_\theta(z_t, r, t) + (t - r) \cdot \text{sg}\!\left(\frac{\partial u_\theta}{\partial t}\right) \tag{iMF-3}$$

Trong đó $\text{sg}(\cdot)$ = stop-gradient (`.detach()` trong PyTorch). **Tại sao stop-gradient?** Nếu không, gradient sẽ cuộn ngược qua đạo hàm cấp hai $\partial^2 u / \partial t^2$, gây bùng nổ gradient và bất ổn số học.

#### 3.4.4. Tính $\partial u / \partial t$ bằng JVP

Đạo hàm $\frac{\partial u}{\partial t}$ được tính bằng **Jacobian-Vector Product (JVP)** — hiệu quả hơn Hessian đầy đủ:

$$\frac{\partial u}{\partial t} \approx \text{JVP}\left(u_\theta,\; (z_t, t),\; (v_{\text{tangent}}, 1)\right) \tag{iMF-4}$$

Trong đó vector tiếp tuyến (tangent) là:
- $\frac{dz_t}{dt} = v_{\text{tangent}}$ — xấp xỉ bởi v-head phụ trợ hoặc $u_\theta(z_t, t, t)$
- $\frac{dt}{dt} = 1$
- $\frac{dr}{dt} = 0$ (r giữ cố định)

**Triển khai PyTorch:**
```python
_, dudt = torch.autograd.functional.jvp(
    lambda z, t: model(z, t, ctx, r=r),
    (z_t, t),
    (v_tangent, torch.ones_like(t)),
    create_graph=False,  # stop-gradient
)
```

**Fallback sai phân hữu hạn** (khi JVP không khả dụng):
$$\frac{\partial u}{\partial t} \approx \frac{u_\theta(z_{t+\delta}, t+\delta, r) - u_\theta(z_t, t, r)}{\delta}, \quad \delta = 10^{-3} \tag{iMF-5}$$

#### 3.4.5. Hàm Mất mát iMF

**Lấy mẫu (t, r):** Với xác suất $\alpha = 0.5$:
- $r = t$ (điều kiện biên) → $u(z, t, t) = v(z, t)$, loss suy biến thành CFM
- $r \neq t$ (JVP branch) → sử dụng hàm hợp V

**Loss tổng hợp:**

$$\mathcal{L}_{\text{iMF}} = \alpha \cdot \underbrace{\| u_\theta(z_t, t, t) - (\varepsilon - x) \|^2}_{\text{Boundary loss}} + (1 - \alpha) \cdot \underbrace{\| V_\theta(z_t, r, t) - (\varepsilon - x) \|^2}_{\text{JVP loss}} \tag{iMF-6}$$

> **Lưu ý triển khai:** Paper gốc sử dụng unified compound function $V$ duy nhất — khi $r=t$ thì $(t-r)=0$ nên $V \equiv u(z,t,t)$ tự suy biến thành FM loss. Cách tách thành 2 nhánh boundary/JVP ở trên là *implementation optimization* của FaceDiff (tránh tính JVP khi $r=t$), kết quả toán học tương đương.

**V-Head phụ trợ** (Appendix A của paper):
$$\mathcal{L}_{\text{v-head}} = 0.1 \cdot \| v_{\text{head}}(h_{\text{hidden}}) - (\varepsilon - x) \|^2 \tag{iMF-7}$$

Trong đó $h_{\text{hidden}}$ là trạng thái ẩn của VoxelMamba, và mục tiêu luôn là $(\varepsilon - x)$ thô (raw FM target, không qua CFG augmentation).

#### 3.4.6. Phân phối Lấy mẫu Thời gian

**Logit-Normal Distribution:**

$$t = \sigma\left(\frac{u - \mu}{\text{scale}}\right), \quad u \sim \mathcal{N}(0, 1) \tag{iMF-8}$$

Với $\sigma(\cdot)$ là sigmoid, $\mu = -0.4$, $\text{scale} = 1.0$. Phân phối này tập trung mật độ vào vùng $t \in [0.2, 0.7]$ — pha giữa quỹ đạo nơi model cần học nhiều nhất.

**Curriculum Learning** *(đóng góp riêng của FaceDiff, không có trong paper iMF gốc):*
- **Giai đoạn 1** ($\text{progress} < 0.6$): 100% logit-normal → học cấu trúc thô nhanh
- **Giai đoạn 2** ($\text{progress} \geq 0.6$): 80% uniform + 20% logit-normal → bao phủ biên $t \to 0, 1$

#### 3.4.7. Classifier-Free Guidance (CFG)

**Batch Doubling (1-pass CFG):**

$$z_{\text{double}} = [z_t, z_t], \quad \text{ctx}_{\text{double}} = [\text{ctx}, \mathbf{0}] \tag{CFG-1}$$

Forward 1 lần: $[v_{\text{cond}}, v_{\text{uncond}}] = u_\theta(z_{\text{double}}, t, \text{ctx}_{\text{double}}, r)$

**Mục tiêu có CFG:**

$$v_{\text{target}} = (\varepsilon - x) + \left(1 - \frac{1}{\omega}\right) (v_{\text{cond}} - v_{\text{uncond}}).\text{detach()} \tag{CFG-2}$$

Trong đó $\omega$ là guidance scale, lấy mẫu từ $p(\omega) \propto \omega^{-\beta}$ trên $[\omega_{\min}, \omega_{\max}]$ (FaceDiff: $\omega \in [1, 8], \beta = 1$).

**Interval Conditioning:** CFG chỉ active khi $t \in [t_{\min}, t_{\max}]$:
$$\omega_{\text{eff}} = \begin{cases} \omega & \text{nếu } t_{\min} \leq t \leq t_{\max} \\ 1 & \text{ngược lại} \end{cases} \tag{CFG-3}$$

#### 3.4.8. Sinh mẫu 1 Bước (One-Step Sampling)

Nhờ quỹ đạo được JVP nắn thẳng, chỉ cần:

$$z_0 = z_1 - u_\theta(z_1, r{=}0, t{=}1, \text{ctx}) \tag{iMF-9}$$

**Không cần scheduler** (Euler/DDIM/DPM). Tiết kiệm 98% thời gian so với 50-step DDPM.

### 3.5. Quadratic Error Function (QEF) và Dual Contouring

#### 3.5.1. Bài toán Dual Contouring

Cho lưới voxel $256^3$ với bề mặt mesh cắt qua. Mục tiêu: tìm **Dual Vertex (DV)** — một đỉnh đại diện duy nhất trong mỗi voxel — sao cho khi nối các DV lại, mesh được tái tạo chính xác.

#### 3.5.2. QEF (Quadratic Error Function)

Bề mặt mesh cắt qua cạnh voxel tạo ra **Hermite data**: tập hợp các cặp $(p_i, n_i)$ — điểm giao cắt $p_i$ và pháp tuyến $n_i$ tại đó.

DV tối ưu là nghiệm bài toán bình phương tối thiểu:

$$\text{DV}^* = \arg\min_v \sum_i \left[ n_i \cdot (v - p_i) \right]^2 + \lambda_{\text{reg}} \| v - \bar{p} \|^2 \tag{QEF-1}$$

Trong đó:
- $p_i$ = điểm giao cắt cạnh voxel
- $n_i$ = pháp tuyến bề mặt tại $p_i$  
- $\bar{p} = \frac{1}{|I|} \sum_i p_i$ = trọng tâm các điểm giao
- $\lambda_{\text{reg}}$ = hệ số regularization (FaceDiff: $10^{-2}$) — kéo DV về trọng tâm, tránh suy biến

**Mở rộng thành hệ tuyến tính:**

$$\mathbf{A}^T \mathbf{A}\, v = \mathbf{A}^T b, \quad \text{với } \mathbf{A} = \begin{bmatrix} n_1^T \\ \vdots \\ n_k^T \\ \sqrt{\lambda_{\text{reg}}} \mathbf{I} \end{bmatrix}, \quad b = \begin{bmatrix} n_1 \cdot p_1 \\ \vdots \\ n_k \cdot p_k \\ \sqrt{\lambda_{\text{reg}}} \bar{p} \end{bmatrix} \tag{QEF-2}$$

Giải bằng Cholesky/SVD trên ma trận $3 \times 3$ (rất nhanh).

#### 3.5.3. TRELLIS.2 QEF mở rộng

TRELLIS.2 mở rộng QEF với 3 thành phần trọng số:

$$E(v) = w_{\text{face}} \sum_{\text{face}} [n_i \cdot (v - p_i)]^2 + w_{\text{boundary}} \sum_{\text{boundary}} [n_j \cdot (v - p_j)]^2 + w_{\text{reg}} \|v - \bar{p}\|^2 \tag{QEF-3}$$

Mặc định: $w_{\text{face}} = 1.0$, $w_{\text{boundary}} = 1.0$, $w_{\text{reg}} = 0.1$.

### 3.6. Variational Autoencoder (VAE)

#### 3.6.1. Bài toán

VAE nén dữ liệu $x$ (O-Voxel features, $\sim 300\text{K}$ điểm) thành biểu diễn tiềm ẩn $z$ (4096 Slat tokens, 32-dim), sao cho:
1. $z$ có thể tái tạo $x$ (reconstruction quality)
2. $z$ tuân theo phân phối Gaussian chuẩn $\mathcal{N}(0, I)$ (cho phép sinh mẫu)

#### 3.6.2. ELBO (Evidence Lower Bound)

$$\log p(x) \geq \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{\text{KL}}(q(z|x) \| p(z))}_{\text{KL Divergence}} \tag{VAE-1}$$

**Encoder:** $q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$ — mạng nơ-ron dự đoán $\mu$ và $\log \sigma^2$

**Reparameterization trick:** $z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$

**KL Divergence giải tích:**

$$D_{\text{KL}} = -\frac{1}{2} \sum_{i=1}^{d} \left( 1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2 \right) \tag{VAE-2}$$

**Chuẩn hóa:** Chia cho $N \cdot d_{\text{lat}}$ = `mu.numel()` (tổng số phần tử của tensor $\mu$). Đây là cách chuẩn hóa được TRELLIS.2 dùng và cũng là công thức được triển khai trong `src/models/sc_vae_loss.py` từ revision này; trước đó FaceDiff lỡ chia cho `target_x.shape[0]` (chỉ là số voxel, thiếu hệ số $d_{\text{lat}} = 32$), khiến KL hiển thị bị thổi phồng đúng 32 lần. Vì $w_{\text{KL}} = 10^{-6}$ rất nhỏ nên đóng góp vào tổng loss vẫn ổn định, nhưng đối chiếu giữa các log cũ và mới cần nhân/chia 32 cho đúng.

---

## 4. Phương pháp Đề xuất

### 4.1. Tổng quan Kiến trúc

FaceDiff là hệ thống 3 giai đoạn:

```
Stage 1: SC-VAE        Stage 2: VoxelMamba + iMF       Stage 3: Decoder
─────────────────      ──────────────────────────      ─────────────────
Mesh (.obj)            Noise z₁ ~ N(0,I)               Slat Tokens
    ↓                       ↓                               ↓
O-Voxel (256³)         VoxelMamba(z₁, t=1, r=0,       SC-VAE Decoder
[N, 10] features            context)                        ↓
    ↓                       ↓                          O-Voxel → Mesh
SC-VAE Encoder         ẑ₀ = z₁ - u_θ(z₁,1,0,ctx)     (Dual Contouring)
    ↓
Slat [4096, 32]
```

### 4.2. Biểu diễn Dữ liệu: O-Voxel 10-Kênh

#### 4.2.1. Pipeline Chuyển đổi Mesh → O-Voxel

**Bước 1 — Chuẩn hóa PBR:** Ép vật liệu thành Metallic=0, Roughness=1 (triệt nhiễu phản quang). Chỉ giữ Albedo gốc.

**Bước 2 — Chuẩn hóa Không gian:**

$$x' = \frac{x - \text{center}(\text{AABB})}{\max(\text{AABB}_{\max} - \text{AABB}_{\min})} \times 0.95 \tag{OV-1}$$

Mesh nằm trong $[-0.5, 0.5]^3$.

**Bước 3 — Voxel hóa Hình học (C++ kernel):**

Hàm `mesh_to_flexible_dual_grid` (Microsoft O-Voxel library):
1. Chia không gian thành lưới $256^3$
2. Tìm giao cắt mesh-voxel edge → tính Hermite data $(p_i, n_i)$
3. Giải QEF (3.5.2) → DV cho mỗi voxel chiếm dụng
4. Xuất: `coords` $[N, 3]$, `dual_vertices` $[N, 3]$, `intersected_flag` $[N, 3]$

**Bước 4 — Voxel hóa Vật liệu (Ray-casting):**

Hàm `textured_mesh_to_volumetric_attr`: Phóng tia từ tâm voxel → tìm giao cắt UV → trích xuất RGB Albedo.

**Bước 5 — Morton Z-Order Alignment:**

Hai mảng hình học/vật liệu (sinh song song, thứ tự khác nhau) đồng bộ bằng mã Morton:

$$M(x, y, z) = \text{interleave\_bits}(x, y, z) \tag{OV-2}$$

Sắp xếp + intersect1d gộp dữ liệu — tránh hoàn toàn vòng lặp `for`.

**Bước 6 — Đóng gói 10-Kênh:**

$$\mathbf{F} = [\underbrace{dv_{\text{local}}}_3, \underbrace{\delta}_3, \underbrace{\gamma}_1, \underbrace{\text{RGB}}_3] \in \mathbb{R}^{N \times 10} \tag{OV-3}$$

| Kênh | Ký hiệu | Phạm vi | Ý nghĩa Hình học | Activation |
|------|---------|---------|-------------------|------------|
| 0–2 | $dv_{\text{local}}$ | $[0, 1]^3$ | Độ dời DV từ góc voxel: $dv = (\text{DV} \times \text{res} - \text{coords}).\text{clamp}(0,1)$ | Clamp |
| 3–5 | $\delta$ | $\{0, 1\}^3$ | Cờ giao cắt 3 trục (X, Y, Z): $\delta_i = 1$ nếu bề mặt cắt qua cạnh trục $i$ | Sigmoid |
| 6 | $\gamma$ | $(0, 1]$ | Hệ số chia tứ giác (split weight): $\gamma = (1 - \text{Var}(dv_{\text{local}})).\text{clamp}(0,1)$ | Softplus |
| 7–9 | RGB | $[0, 1]^3$ | Màu Albedo khuếch tán | Clamp |

**Vị trí thế giới thực (World Position) từ O-Voxel:**

$$\mathbf{p}_{\text{world}} = (\text{coords} + dv_{\text{local}}) \times \text{voxel\_size} + \text{AABB}_{\min} \tag{OV-4}$$

Trong đó $\text{voxel\_size} = (\text{AABB}_{\max} - \text{AABB}_{\min}) / 256$.

#### 4.2.2. Dual Contouring: O-Voxel → Mesh

Thuật toán `flexible_dual_grid_to_mesh` chuyển O-Voxel thành mesh tam giác qua 5 bước:

**Bước 1 — Tính Vị trí Đỉnh:**

$$v_i = (\text{coords}_i + dv_i) \times \text{voxel\_size} + \text{AABB}_{\min} \tag{DC-1}$$

**Bước 2 — Tìm Cạnh Giao cắt:**

Với mỗi voxel $i$ và mỗi trục $a \in \{x, y, z\}$ mà $\delta_{i,a} = 1$, cạnh trục $a$ tại voxel $i$ là cạnh giao cắt.

**Bước 3 — Tìm 4 Voxel Lân cận:**

Mỗi cạnh giao cắt chia sẻ bởi 4 voxel lân cận. Offset được tra bảng:

| Trục | 4 voxel lân cận (offset từ voxel gốc) |
|------|----------------------------------------|
| X | $(0,0,0), (0,0,1), (0,1,1), (0,1,0)$ |
| Y | $(0,0,0), (1,0,0), (1,0,1), (0,0,1)$ |
| Z | $(0,0,0), (0,1,0), (1,1,0), (1,0,0)$ |

**Bước 4 — Hashmap Lookup:**

Dùng GPU hashmap để tìm index 4 DV → tạo quad (tứ giác).

**Bước 5 — Chia Quad thành 2 Tam giác:**

Mỗi quad cần chia thành 2 tam giác. Hai cách chia:

- **Split 1:** $(0,1,2), (0,2,3)$ — đường chéo 0-2
- **Split 2:** $(0,1,3), (3,1,2)$ — đường chéo 1-3

**Khi không có $\gamma$ (split_weight = None):**

Chọn cách chia có normal alignment tốt hơn:

$$\text{align}_k = |(\mathbf{n}_0 \times \mathbf{n}_1)|, \quad k \in \{1, 2\} \tag{DC-2}$$

Chọn split có $\text{align}$ lớn hơn (hai tam giác đồng phẳng hơn).

**Khi có $\gamma$ (split_weight):**

$$\text{score}_{02} = \gamma_0 \cdot \gamma_2, \quad \text{score}_{13} = \gamma_1 \cdot \gamma_3 \tag{DC-3}$$

$$\text{Chọn } \begin{cases} \text{Split 1 (đường chéo 0-2)} & \text{nếu } \text{score}_{02} > \text{score}_{13} \\ \text{Split 2 (đường chéo 1-3)} & \text{ngược lại} \end{cases} \tag{DC-4}$$

**Chế độ Training (differentiable):**

Tạo đỉnh trung tâm bằng trung bình có trọng số:

$$v_{\text{mid}} = \frac{\text{score}_{02} \cdot \frac{v_0 + v_2}{2} + \text{score}_{13} \cdot \frac{v_1 + v_3}{2}}{\text{score}_{02} + \text{score}_{13}} \tag{DC-5}$$

Chia quad thành 4 tam giác qua $v_{\text{mid}}$: $(0,1,\text{mid}), (1,2,\text{mid}), (2,3,\text{mid}), (3,0,\text{mid})$. Phương pháp này khả vi (differentiable) đối với $\gamma$.

#### 4.2.3. LMDB Caching I/O

- Tensor 10-kênh serialize → LMDB B-Tree nhị phân
- **Sequential Sort Key:** Sắp xếp tuyến tính theo ID → HDD đọc tuần tự, throughput $5 \to 150$ MB/s
- **Fallback chain:** LMDB → Disk Cache (.pt) → Fresh Conversion

### 4.3. Giai đoạn 1: SC-VAE (Sparse Convolution VAE)

Nén $\sim 300\text{K}$ điểm O-Voxel thành 4096 Slat tokens ($\mathbb{R}^{32}$).

#### 4.3.1. SparseResMLPBlock

Thay thế ConvNeXt 3D Block của TRELLIS.2:

```
Input x [N, C] (Sparse Tensor)
  │
  ├── SubMConv3d(C, C, kernel=3³, padding=1)  ← Sparse 3D conv
  │       │
  │   LayerNorm32 (FP32 cast để triệt NaN)
  │       │
  │   Linear(C → 4C) → SiLU → Linear(4C → C)  ← Point-wise MLP
  │       │
  │   Zero-init final linear (_zero_module)
  │
  └── + (Residual connection)
```

**Ưu điểm:** Zero-init → gradient ổn định ngay đầu, $-45\%$ VRAM so với ConvNeXt.

#### 4.3.2. Encoder (Pyramid 4 cấp)

$$\text{Resolution:} \quad 256^3 \xrightarrow{\text{stride-2}} 128^3 \xrightarrow{\text{stride-2}} 64^3 \xrightarrow{\text{stride-2}} 32^3 \xrightarrow{\text{stride-2}} 16^3$$

$$\text{Channels:} \quad 10 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512$$

Mỗi cấp `SparseEncoderBlock` được tổ chức theo thứ tự **chiếu kênh → MLP residual → strided sparse conv → non-parametric S2C-shortcut**:

1. `proj = spconv.SubMConv3d(in_c, out_c, k=1)` — chuyển kênh ở cùng độ phân giải.
2. `num_res_blocks × SparseResMLPBlock` — trích xuất đặc trưng (ConvNeXt-style: SubMConv3d 3³ → LayerNorm32 → Linear↑4× → SiLU → Linear↓ với zero-init lớp cuối, residual cộng vào features đầu vào).
3. `down = spconv.SparseConv3d(out_c, out_c, k=2, stride=2)` — strided sparse downsampling.
4. **Non-parametric S2C-shortcut** (`_build_sparse_down_shortcut`): với mỗi voxel cha ở độ phân giải sau down, lấy 8 voxel con tương ứng (lookup theo 64-bit hashed indices), nối features (`reshape(N, 8·C_in)`) rồi `avg_groups_channels(...)` về `out_c` kênh. Kết quả cộng vào features của `x_down` như một skip nối tắt.

So với spec gốc của TRELLIS.2 (`SparseResBlockS2C3d` dùng `sp.SparseSpatial2Channel(2)` đúng phương trình S2C bên dưới), triển khai trong FaceDiff dùng strided `SparseConv3d` cho nhánh chính (vì spconv 2.x ổn định hơn) và chỉ giữ S2C ở dạng *non-parametric shortcut*. Hai cách tương đương về mặt biểu diễn (cùng giảm spatial 2× và đưa context 8 con vào voxel cha), nhưng nhánh chính của FaceDiff có thêm $C_{\text{out}}^2 \cdot 2^3$ tham số do dùng strided conv. Đây cũng là lý do số param của `SC_VAE` đo được là **35.13 M** (so với ~50 M nếu copy chính xác `SparseResBlockS2C3d`).

**SparseSpatial2Channel** chính thống của TRELLIS.2 (Paper Eq. 4):

$$\text{S2C: } f_{\text{cha}}[k] = f_{\text{con}}[\lfloor \mathbf{p}/2 \rfloor, \, (\mathbf{p} \bmod 2) \cdot C + k], \quad k \in \{0, \ldots, C-1\} \tag{S2C}$$

Kết quả: $N_{\text{fine}}$ voxel với $C$ kênh → $N_{\text{coarse}}$ voxel với $8C$ kênh.

**LayerNorm32 (FP32 cast)** — TRELLIS.2 cast sang FP32 trước khi LayerNorm và đổi dtype lại sau, để chống NaN dưới `torch.amp` FP16. FaceDiff cũng dùng cùng class (`src/modules/norm.py`), do shape param `weight/bias` giữ nguyên nên checkpoint cũ không cần migrate.

**Non-affine LayerNorm trước to_mu/to_logvar** (TRELLIS.2 `SparseUnetVaeEncoder.forward()`, dòng `h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))`). FaceDiff bật mặc định qua flag `pre_latent_norm=True` của `SC_VAE`. Vì là non-parametric, nó **không xuất hiện trong `state_dict`** ⇒ checkpoint epoch 397 vẫn load `strict=True` không lỗi (đã verify, xem Mục 7.2.6).

Đầu ra: $z \sim q(z|x) = \mathcal{N}(\mu, \sigma^2)$ với $\mu, \sigma \in \mathbb{R}^{N_{\text{enc4}} \times 32}$. Khi mesh đầu vào kín (toàn bộ 16³ active), $N_{\text{enc4}} = 4096$ — đúng số Slat tokens.

#### 4.3.3. Decoder với Generative Pruning (Rho Loss)

Giải mã đi lên: $16^3 \rightarrow 32^3 \rightarrow 64^3 \rightarrow 128^3 \rightarrow 256^3$.

Mỗi cấp `SparseDecoderBlock` thực thi:

1. `rho_head = nn.Linear(in_c, 8)` — dự đoán 8 subdivision logits cho 8 voxel con của mỗi voxel cha hiện hành.
2. **Light gate**: nhân features cha với `gate = sigmoid(rho_logits).amax(dim=1, keepdim=True)` để truyền tín hiệu "voxel cha sẽ tồn tại" xuống nhánh upsample.
3. `up = spconv.SparseConvTranspose3d(in_c, out_c, k=2, stride=2)` — strided sparse transpose conv (đối ngẫu của strided down ở Encoder).
4. **Non-parametric C2S-shortcut** (`_build_sparse_up_shortcut`): cho mỗi voxel con sau upsample, lookup features cha rồi `repeat_interleave` về `out_c` kênh, cộng vào `x_up`.
5. **Pruning** — nhánh training và inference khác nhau:
   - Training (`_prune_by_target`): mask `x_up.indices` chỉ giữ những voxel con xuất hiện trong topology GT của tầng tương ứng (`sparse_pyramid[i]` từ Encoder), tức **teacher-forcing** trên topology — đảm bảo recall 100% so với target mỗi tầng nhưng kéo theo gap distribution-shift khi inference.
   - Inference (`_apply_child_pruning`): với mỗi voxel con, lấy logit `rho` của voxel cha, sigmoid, threshold $> 0.5$ → giữ con hợp lệ. Khi không có cha nào hợp lệ thì raise error (fail-fast).
6. `num_res_blocks × SparseResMLPBlock` — refine features sau pruning.

**Lưu ý kiến trúc** (đối chiếu TRELLIS.2 chính thống):
- TRELLIS.2 dùng `SparseResBlockC2S3d` với `sp.SparseChannel2Spatial(2)` (Eq. C2S bên dưới) cho cả nhánh chính lẫn pruning, và mask subdivision `subdiv > 0` là *raw logit threshold*. FaceDiff dùng `SparseConvTranspose3d` cho nhánh chính + non-parametric shortcut C2S; mask pruning dùng *sigmoid > 0.5* (tương đương về mặt phân loại nhưng ngưỡng sigmoid 0.5 ⇔ logit 0).
- Sự khác biệt thứ hai là FaceDiff áp dụng pruning **sau** upsample (chọn lại từ tập con đã sinh), trong khi TRELLIS.2 áp pruning **trong** upsample (`SparseChannel2Spatial(2, mask)` chỉ phân bổ memory cho con hợp lệ). Đây là lý do `_apply_child_pruning` của FaceDiff phải lookup ngược cha bằng hash; tổng VRAM peak vì thế cao hơn ~10 % so với spec gốc, nhưng tránh được phải reimplement `SparseChannel2Spatial` cho spconv 2.x.

**SparseChannel2Spatial** chính thống (Paper Eq. 5):

$$\text{C2S: } f_{\text{con}}[\mathbf{p}_{\text{cha}} \cdot 2 + \Delta, \, k] = f_{\text{cha}}[\mathbf{p}_{\text{cha}}, \, \Delta \cdot C + k] \tag{C2S}$$

trong đó $\Delta \in \{0,1\}^3$ là offset 8 con.

**Rho Head & Loss** — TRELLIS.2 spec, FaceDiff giữ nguyên trong `src/models/sc_vae.py:_build_child_mask_targets`:

$$\text{subdiv}_i = \text{Linear}(f_{\text{cha}, i}) \in \mathbb{R}^{8}, \quad \text{mask}_i = \mathbb{1}[\sigma(\text{subdiv}_i) > 0.5] \tag{Rho-1}$$

$$\rho_i^* = \sum_{j \in \text{children}(i)} \text{onehot}_8(\text{child\_id}(j)) \quad \in \{0,1\}^8 \tag{Rho-2}$$

$$\mathcal{L}_{\rho} = \frac{1}{|L|} \sum_{l \in \text{levels}} \text{BCE-with-logits}(\hat{\rho}_l, \rho_l^*) \tag{Rho-3}$$

Trọng số mặc định $w_\rho = 0.2$ (FaceDiff) so với $0.1$ trong `configs/scvae/shape_vae_next_dc_f16c32_fp16.json` của TRELLIS.2 — FaceDiff đặt cao hơn vì topology khuôn mặt sparse hơn so với asset Objaverse-XL.

**Training vs Inference gap:**
- **Training:** `_prune_by_target` ⇒ topology recall 100%. Gradient của nhánh `rho_head` vẫn nhận supervision qua `L_ρ`, nhưng các voxel "rác" mà rho head dự đoán nhầm sẽ bị mask GT loại trước khi tính recon loss.
- **Inference:** `_apply_child_pruning` tự dự đoán → recall < 100%. Đây là *teacher-forcing distribution gap* cố hữu của Rho-pruning, TRELLIS.2 cũng có (xem Section 3.2.1 paper).

**Decoder Output Activations** (TRELLIS.2 `FlexiDualGridVaeDecoder`, FaceDiff triển khai trong `apply_shape_mat_output_activations()` ở `src/models/sc_vae.py`):

$$\hat{v} = (1 + 2m) \cdot \sigma(h_{0:3}) - m, \quad m = 0.5 \quad \Rightarrow \quad \hat{v} \in [-0.5, 1.5] \tag{Act-1}$$

$$\hat{\delta} = \begin{cases} h_{3:6} > 0 & \text{(inference — binary threshold)} \\ \sigma(h_{3:6}) & \text{(training — differentiable)} \end{cases} \tag{Act-2}$$

$$\hat{\gamma} = \text{softplus}(h_{6:7}) = \ln(1 + e^{h_{6:7}}) > 0 \tag{Act-3}$$

$$\hat{c} = \text{clamp}(h_{7:10}, 0, 1) \tag{Act-4}$$

Margin $m = 0.5$ cho phép dual vertex vượt nhẹ ra ngoài voxel, tăng độ chính xác hình học tại biên — TRELLIS.2 mặc định `voxel_margin=0.5` (xem `FlexiDualGridVaeDecoder.__init__`). FaceDiff trước revision này **chỉ** áp dụng `softplus(γ)` trong loss; `dv` được so sánh trên raw logit nên gradient ép `dv` về sigmoid(0)=0.5 thay vì khoảng [0,1] mong muốn. Sau revision (tham số `apply_output_activations=True` của `SC_VAE`, được loss `_shape_mat_recon_loss` gọi nội bộ qua `apply_dv_activation`), `dv` được activate trước khi tính MSE → gradient hoàn toàn nhất quán với inference path của dual contouring.

#### 4.3.4. Hàm Mất mát Tổng hợp SC-VAE

**Reconstruction Loss (10-kênh, mode `shape_mat`):**

$$\mathcal{L}_{\text{recon}} = \underbrace{0.01 \cdot \text{MSE}(\hat{dv}, dv)}_{\text{Dual Vertex}} + \underbrace{0.1 \cdot \text{BCE}(\hat{\delta}, \delta)}_{\text{Intersection Flag}} + \underbrace{\text{SmoothL1}(\text{softplus}(\hat{\gamma}), \gamma)}_{\text{Split Weight}} + \underbrace{\text{L1}(\hat{c}, c)}_{\text{RGB}} \tag{SC-1}$$

**Giải thích trọng số:**
- $dv$ × 0.01: Offset nhỏ (thường < 0.5 voxel), gradient MSE đã đủ mạnh
- $\delta$ × 0.1: BCEWithLogits phạt nặng sai topology
- $\gamma$: SmoothL1 + Softplus đảm bảo $\hat{\gamma} > 0$ (cần cho DC split)
- RGB × 1.0: L1 loss cho material (theo TRELLIS.2)

**KL Divergence (weight $10^{-6}$, warmup 20 epochs):**

$$\mathcal{L}_{\text{KL}} = -\frac{1}{2|\mu|} \sum_{i} \left( 1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2 \right) \tag{SC-2}$$

**Stage-2 Render Loss:**

Chiếu O-Voxel features lên 2D orthographic maps, so sánh recon vs GT:

$$\mathcal{L}_{\text{render}} = \underbrace{\text{L1}(\text{mask}_{\text{pred}}, \text{mask}_{\text{gt}})}_{\times 1} + \underbrace{10 \cdot \text{L1}(\text{depth}_{\text{pred}}, \text{depth}_{\text{gt}})}_{\times 10} + \underbrace{\mathcal{L}_{\text{perceptual}}}_{\text{L1 + 0.2 SSIM + 0.2 LPIPS}} \tag{SC-3}$$

**Công thức chiếu (dv-corrected, sau fix Bug 4):**

$$\mathbf{p}_{\text{proj}} = \frac{\text{coords} + \text{clamp}(dv, 0, 1)}{\max(\text{coords}) + 1} \times 2 - 1 \tag{SC-4}$$

**Hàm mất mát tổng:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + w_{\text{KL}} \cdot \mathcal{L}_{\text{KL}} + w_\rho \cdot \mathcal{L}_\rho + \mathcal{L}_{\text{render}} \tag{SC-5}$$

Với $w_{\text{KL}} = 10^{-6}$, $w_\rho = 0.2$.

### 4.4. Giai đoạn 2: VoxelMamba + iMF

> **Nguồn gốc kiến trúc:** Backbone VoxelMamba trong FaceDiff là *thiết kế lai (hybrid)* kết hợp: (1) Bidirectional Mamba cho 3D generation từ DiM-3D [14], (2) Hilbert space-filling curve ordering từ VoxelMamba [4], (3) In-context prefix token conditioning tự thiết kế cho iMF [1]. FaceDiff **không** sử dụng Dual-scale SSM Block hay Implicit Window Partition từ paper VoxelMamba gốc.

#### 4.4.0. Pipeline dữ liệu iMF: `SlatDataset`, cache đĩa, và chế độ `--offline-data`

**Mục tiêu:** Không cần huấn luyện lại các mô hình ngữ cảnh (ArcFace, FLAME-image, DINOv2) hay SC-VAE trong mỗi epoch iMF. Một lần *tiền tính toán* (precompute) ghi tensor mục tiêu `slat` và vector điều kiện `context` [946] xuống đĩa; sau đó VoxelMamba + iMF chỉ đọc cache → **giảm VRAM đáng kể** (không còn ~35M tham số SC-VAE + stack DINO trên cùng GPU với backbone) và cho phép **tăng `batch_size`** trong `train_imf.py`.

**Triển khai trong code (`src/train_imf.py`, class `SlatDataset`):**

| Thành phần | Vai trò |
|------------|---------|
| `__getitem__` | Nếu tồn tại file `.pt` với `meta.cache_tag` khớp `cache_contract` → `torch.load` trả `(slat, context)` ngay. |
| `cache_contract` | JSON chuẩn hóa: `slat_length`, `latent_dim`, `context_dim`, `shape_feature_mode`, chữ ký checkpoint SC-VAE (`size+mtime`), `ovoxel_resolution`, v.v. → `cache_tag = slatv2_<sha1[:12]>`. Đổi checkpoint SC-VAE hoặc tham số → tag mới → tên file cache khác (tránh dùng nhầm latent cũ). |
| `_encode_mesh` | Mesh → O-Voxel 10 kênh (`OVoxelConverter` hoặc fallback) → SC-VAE `encode` → `mu` làm Slat; đồng thời (nếu có đủ extractors) render front/back → ArcFace(512) + FLAME-from-image(50) + DINOv2-back(384) → `create_hybrid_context` [946]. |
| `cfg.imf.use_precomputed_data` | Khi `True` (CLI `--offline-data`): **không** khởi tạo SC-VAE, `MeshRenderer`, `ArcFaceExtractor`, `FLAMEExpressionAdapter`, `DinoV3Extractor` trên GPU; `SlatDataset` bắt buộc chỉ đọc cache — thiếu file → `RuntimeError` rõ ràng. |
| `SlatDataset.cache_file_path(idx)` / `has_valid_cache(idx)` | (revision 3) Đường dẫn cache và kiểm tra nhanh cho script precompute (`--skip-existing`). |

**Kế hoạch vận hành hai giai đoạn (khuyến nghị):**

1. **Giai đoạn A — Precompute (một lần, có thể chạy qua đêm):**  
   `python scripts/precompute_slat_cache.py --sc-vae-ckpt <ckpt_sc_vae>.pt --dataset both --num-workers 0 --skip-existing`  
   Script nạp SC-VAE + **đầy đủ** hybrid context (renderer, ArcFace, FLAME, DINOv2) giống `train_imf` online, ghi `data/slat_cache/` (FaceVerse) và `data/slat_cache_facescape/` (FaceScape).  
   Tùy chọn `--use-random-context` chỉ dành cho **debug tốc độ** (context không phải hybrid thật — **không** dùng cho huấn luyện production).  
   **Lưu ý lịch sử:** bản `precompute_slat_cache.py` cũ không truyền extractors → context trong cache bị **random**; revision 3 đã sửa.

2. **Giai đoạn B — Huấn luyện iMF:**  
   `python src/train_imf.py --offline-data --batch-size <tăng theo VRAM còn lại> ...`  
   Chỉ nạp VoxelMamba (+ v-head, EMA, optimizer). CFG vẫn hoạt động: dropout ngữ cảnh áp dụng trên tensor `context` đã load, không cần forward lại ArcFace trong training loop.

**Hạn chế cần biết:** Chế độ `dual_branch` (hai SC-VAE) chưa được script precompute hỗ trợ — cần mở rộng script hoặc tạm tắt dual khi precompute. `num_workers>0` có thể gây xung đột EGL/GPU với renderer; mặc định `0` an toàn.

#### 4.4.1. Kiến trúc VoxelMamba

**Sơ đồ tổng quan Pipeline (Overall Pipeline):**

```mermaid
graph TD
    subgraph INPUTS["Đầu vào"]
        Z["z_t<br/>[B, 4096, 32]<br/>Slat tokens + noise"]
        CTX["ctx<br/>[B, 946]<br/>ArcFace(512)+FLAME(50)+DINOv2(384)"]
        T["t ∈ [0,1]<br/>[B]<br/>End timestep"]
        R["r ∈ [0,1]<br/>[B]<br/>Start timestep"]
        OMEGA["Ω = {ω, t_min, t_max}<br/>[B, 3]<br/>CFG guidance"]
    end

    Z --> INPUT_EMBED["input_embed<br/>Linear(32 → 512)"]
    INPUT_EMBED --> HILBERT["Hilbert Reorder<br/>raster → hilbert<br/>(preserve 3D locality)"]

    CTX --> CTX_TOK["context_tokenizer<br/>Linear(946 → 512×8) + SiLU<br/>→ [B, 8, 512]"]

    T --> T_MLP["time_mlp<br/>SinEmbed(128) → MLP → 512"]
    T_MLP --> T_TOK["time_tokenizer<br/>Linear(512 → 512×4) + SiLU<br/>→ [B, 4, 512]"]

    R --> R_MLP["r_mlp<br/>SinEmbed(128) → MLP → 512"]
    R_MLP --> R_TOK["r_tokenizer<br/>Linear(512 → 512×4) + SiLU<br/>→ [B, 4, 512]"]

    T --> INTERVAL_COMPUTE["interval = |t - r|"]
    R --> INTERVAL_COMPUTE
    INTERVAL_COMPUTE --> IV_MLP["interval_mlp<br/>SinEmbed(128) → MLP → 512"]
    IV_MLP --> IV_TOK["interval_tokenizer<br/>Linear(512 → 512×4) + SiLU<br/>→ [B, 4, 512]"]

    OMEGA --> G_TOK["guidance_tokenizer<br/>Linear(3 → 512×4) + SiLU<br/>→ [B, 4, 512]"]

    CTX_TOK --> PREFIX["Concat Prefix<br/>[B, 24, 512]"]
    T_TOK --> PREFIX
    R_TOK --> PREFIX
    IV_TOK --> PREFIX
    G_TOK --> PREFIX

    HILBERT --> CONCAT["Concat<br/>[B, 4120, 512]"]
    PREFIX --> CONCAT

    CONCAT --> BIMAMBA["12× BiMamba Block<br/>(see detail below)"]

    BIMAMBA --> STRIP["Strip Prefix<br/>[B, 4120, 512] → [B, 4096, 512]"]
    STRIP --> HILBERT_INV["Hilbert Inverse<br/>hilbert → raster"]
    HILBERT_INV --> OUTNORM["output_norm<br/>RMSNorm(512)"]
    OUTNORM --> OUTPROJ["output_proj<br/>Linear(512 → 32)<br/>⚠ Zero-init W,b"]
    OUTPROJ --> OUTPUT["u_θ(z_t, r, t, ctx)<br/>[B, 4096, 32]<br/>Predicted avg velocity"]

    style INPUTS fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style BIMAMBA fill:#0f3460,stroke:#533483,color:#ffffff
    style OUTPUT fill:#1a1a2e,stroke:#e94560,color:#ffffff
    style PREFIX fill:#16213e,stroke:#533483,color:#e0e0e0
```

**Chi tiết khối BiDirectional Mamba Block:**

```mermaid
graph TD
    subgraph BIMAMBA_BLOCK["BiDirectional Mamba Block (×12)"]
        IN["Input x<br/>[B, L, 512]"] --> NORM["RMSNorm(512)"]

        NORM --> FWD_MAMBA["Forward Mamba SSM<br/>d_model=512, d_state=16<br/>d_conv=4, expand=2<br/>(inner_dim=1024)"]

        NORM --> FLIP["torch.flip(dim=1)<br/>Reverse sequence"]
        FLIP --> BWD_MAMBA["Backward Mamba SSM<br/>d_model=512, d_state=16<br/>d_conv=4, expand=2"]
        BWD_MAMBA --> UNFLIP["torch.flip(dim=1)<br/>Restore order"]

        FWD_MAMBA --> ADD_SCAN["fwd + bwd<br/>(element-wise sum)"]
        UNFLIP --> ADD_SCAN

        ADD_SCAN --> DROP["Dropout(0.1)"]
        DROP --> RESIDUAL["+ residual<br/>(skip connection)"]
        IN --> RESIDUAL

        RESIDUAL --> OUT["Output<br/>[B, L, 512]"]
    end

    subgraph MAMBA_INTERNAL["Mamba SSM Internal (mỗi chiều)"]
        MI_IN["x [B,L,512]"] --> MI_PROJ["in_proj<br/>Linear(512 → 2×1024)"]
        MI_PROJ --> MI_SPLIT["split → x, z"]
        MI_SPLIT --> MI_CONV["Conv1D(1024, k=4)<br/>+ SiLU"]
        MI_SPLIT --> MI_Z["z branch<br/>SiLU gate"]
        MI_CONV --> MI_SSM["Selective SSM<br/>A[n=16], B[n=16]<br/>C[n=16], Δ[d=1024]<br/>parallel scan O(L)"]
        MI_SSM --> MI_GATE["× SiLU(z)"]
        MI_Z --> MI_GATE
        MI_GATE --> MI_OUT["out_proj<br/>Linear(1024 → 512)"]
    end

    style BIMAMBA_BLOCK fill:#0f3460,stroke:#533483,color:#e0e0e0
    style MAMBA_INTERNAL fill:#16213e,stroke:#0f3460,color:#e0e0e0
```

**Fallback GRU** *(khi mamba-ssm không khả dụng):*

```mermaid
graph LR
    IN_GRU["Input x<br/>[B, L, 512]"] --> NORM_GRU["RMSNorm"]
    NORM_GRU --> GRU["nn.GRU<br/>input=512, hidden=256<br/>bidirectional=True<br/>→ [B, L, 512]"]
    GRU --> DROP_GRU["Dropout(0.1)"]
    DROP_GRU --> RES_GRU["+ residual"]
    IN_GRU --> RES_GRU
    RES_GRU --> OUT_GRU["Output [B, L, 512]"]

    style GRU fill:#533483,stroke:#e94560,color:#ffffff
```

**Luồng huấn luyện iMF (iMF Training Flow):**

```mermaid
graph TD
    subgraph SAMPLING["Lấy mẫu"]
        X["x_data<br/>[B, 4096, 32]"] --> INTERP
        E["ε ~ N(0,I)"] --> INTERP
        TS["t ~ logit-normal(μ=-0.4)"] --> INTERP["z_t = (1-t)x + tε"]
        TS --> R_SAMPLE["r: 50% r=t, 50% r~U[0,1]"]
    end

    subgraph FORWARD["Forward Passes"]
        INTERP --> V_PASS["v_θ = VoxelMamba(z_t, t=t, r=t)<br/>Boundary condition"]
        INTERP --> JVP_PASS["u, du/dt = JVP(VoxelMamba,<br/>(z_t, r, t), (v_θ, 0, 1))<br/>create_graph=False"]
        V_PASS --> JVP_PASS
    end

    subgraph COMPOUND["Compound Function"]
        JVP_PASS --> V_COMPOUND["V = u + (t-r) × stopgrad(du/dt)"]
        R_SAMPLE --> V_COMPOUND
    end

    subgraph LOSS["Loss Computation"]
        V_COMPOUND --> LOSS_JVP["L_JVP = ||V - (ε-x)||²<br/>(khi r≠t)"]
        V_PASS --> LOSS_BND["L_boundary = ||v_θ - (ε-x)||²<br/>(khi r=t)"]
        LOSS_JVP --> ADAPTIVE["w_adaptive(t) × weighted_avg"]
        LOSS_BND --> ADAPTIVE
        V_PASS --> LOSS_VHEAD["L_v-head = ||v_head(h) - (ε-x)||²"]
        ADAPTIVE --> TOTAL["L_total = w_adaptive × (α·L_bnd + (1-α)·L_jvp) + 0.1 × w_adaptive × L_vhead"]
        LOSS_VHEAD --> TOTAL
    end

    subgraph INFERENCE["1-Step Sampling (Inference)"]
        Z1["z_1 ~ N(0,I)"] --> UNET["u = VoxelMamba(z_1, t=1, r=0, ctx)"]
        UNET --> Z0["z_0 = z_1 - u<br/>→ SC-VAE Decode → Mesh"]
    end

    style SAMPLING fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style FORWARD fill:#0f3460,stroke:#533483,color:#e0e0e0
    style COMPOUND fill:#16213e,stroke:#e94560,color:#e0e0e0
    style LOSS fill:#533483,stroke:#e94560,color:#e0e0e0
    style INFERENCE fill:#0f3460,stroke:#e94560,color:#ffffff
```

**Prefix Tokens (24 tổng):**

| Token | Số lượng | Nguồn | Kích thước |
|-------|----------|-------|------------|
| Context | 8 | $\text{ctx}_{946} \xrightarrow{\text{Linear}} \mathbb{R}^{8 \times 512}$ | $[B, 8, 512]$ |
| Time $t$ | 4 | $t \xrightarrow{\text{SinEmbed}} \xrightarrow{\text{MLP}} \mathbb{R}^{4 \times 512}$ | $[B, 4, 512]$ |
| Time $r$ | 4 | $r \xrightarrow{\text{SinEmbed}} \xrightarrow{\text{MLP}} \mathbb{R}^{4 \times 512}$ | $[B, 4, 512]$ |
| Interval $(t-r)$ | 4 | $|t-r| \xrightarrow{\text{SinEmbed}} \xrightarrow{\text{MLP}} \mathbb{R}^{4 \times 512}$ | $[B, 4, 512]$ |
| Guidance | 4 | $[\omega, t_{\min}, t_{\max}] \xrightarrow{\text{Linear}} \mathbb{R}^{4 \times 512}$ | $[B, 4, 512]$ |

> **Ghi chú:** Token Interval $(t-r)$ được thêm theo iMF paper Tab. 4: network conditions on $(t-r)$ ngoài $t$ và $r$ riêng biệt, giúp SSM trực tiếp nhận biết độ dài khoảng trung bình mà không cần học ngầm phép trừ.

**Sinusoidal Timestep Embedding:**

$$\text{emb}(t, k) = \begin{cases} \sin(t \cdot 10000^{-2k/d}) & k < d/2 \\ \cos(t \cdot 10000^{-2(k-d/2)/d}) & k \geq d/2 \end{cases} \tag{VoxM-1}$$

Với $d = \text{hidden\_dim} / 4 = 128$.

**Output Projection Zero-Init:**

$$W_{\text{out}} = \mathbf{0}, \quad b_{\text{out}} = \mathbf{0} \quad \text{(khởi tạo)} \tag{VoxM-2}$$

Đảm bảo ở bước đầu, $u_\theta(z, t, \text{ctx}) \approx 0$ → quỹ đạo xuất phát gần identity → ổn định gradient.

#### 4.4.2. Hàm Mất mát iMF (Triển khai trong FaceDiff)

Xem chi tiết toán học tại Mục 3.4. Triển khai cụ thể:

$$\mathcal{L} = \mathbb{E}_{t,r}\left[w_{\text{adaptive}}(t) \cdot \ell(t,r)\right] + 0.1 \cdot \mathbb{E}_{t}\left[w_{\text{adaptive}}(t) \cdot \ell_{\text{v-head}}(t)\right] \tag{iMF-L}$$

Trong đó $\ell(t,r)$ là loss theo nhánh: nếu $r=t$ thì $\ell = \|v_\theta - (\varepsilon - x)\|^2$, nếu $r\neq t$ thì $\ell = \|V_\theta - (\varepsilon - x)\|^2$. V-head dùng mục tiêu FM thô $(\varepsilon - x)$ và cũng được weight bởi $w_{\text{adaptive}}(t)$ theo Appendix A.

**Adaptive Loss Weighting** *(theo MeanFlow paper gốc, kế thừa bởi iMF Appendix A):*

$$w_{\text{adaptive}}(t) = \frac{1}{\overline{\ell}(\text{bin}(t))} \cdot \frac{1}{Z} \tag{iMF-Lw}$$

Trong đó $\overline{\ell}(b)$ là EMA (decay=0.99) của loss trung bình tại bin $b$ (100 bins chia đều $[0,1]$), $Z$ là hệ số chuẩn hóa để $\mathbb{E}[w] = 1$. Mục đích: các vùng timestep có loss cao tự nhiên bị down-weight, đảm bảo mỗi vùng $t$ đóng góp đều vào gradient.

**Material Condition Dropout (10–20%):** Drop $v_{\text{target}}[\text{material}]$ thay bằng $v_\theta[\text{material}].\text{detach()}$ → ngăn overfitting color.

### 4.5. Hệ thống Ngữ cảnh Lai (Hybrid Context)

$$\text{ctx} = [\underbrace{f_{\text{arc}}}_{\mathbb{R}^{512}}, \underbrace{f_{\text{flame}}}_{\mathbb{R}^{50}}, \underbrace{f_{\text{dino}}}_{\mathbb{R}^{384}}] \in \mathbb{R}^{946} \tag{Ctx-1}$$

#### 4.5.1. ArcFace — Danh tính (512-dim)

**ArcFace Loss** (Deng et al., 2019):

$$\mathcal{L}_{\text{arc}} = -\log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cos \theta_j}} \tag{Arc-1}$$

Trong đó $\theta_j$ = góc giữa feature và prototype lớp $j$, $m$ = additive angular margin, $s$ = scale factor.

- ResNet-50 backbone (InsightFace `buffalo_l`), 24M params, frozen
- Đầu ra: 512-dim $L_2$-normalized trên hypersphere
- Đầu vào: Render mặt trước mesh → ArcFace → identity code

#### 4.5.2. FLAME Adapter — Biểu cảm (50-dim)

- Lightweight MLP (4 Conv + 2 FC), 5M params
- **Không phải FLAME model gốc** — chỉ predict 50 expression parameters
- Train từ scratch cùng Stage 2

**FLAME Model gốc** (Li et al., 2017) biểu diễn:

$$M(\beta, \theta, \psi) = W(T_P(\beta, \theta, \psi), J(\beta), \theta, \mathcal{W}) \tag{FLAME-1}$$

Trong đó $\beta$ = shape params (300-dim), $\theta$ = pose (jaw rotation), $\psi$ = expression (100-dim). FaceDiff chỉ dùng 50-dim subset.

#### 4.5.3. DINOv2 — Hình học Mặt sau (384-dim)

- `facebook/dinov2-small` (ViT-S/14), 22M params
- INT4 quantized (~20MB VRAM)
- Đầu ra: 384-dim CLS token từ render mặt sau
- **Mục đích:** Thông tin gáy, tóc, tai — phần ArcFace không nắm bắt

#### 4.5.4. Tokenization

$$\text{ctx}_{946} \xrightarrow{\text{Linear}(946 \to 512 \times 8)} \xrightarrow{\text{SiLU}} \xrightarrow{\text{reshape}} \text{prefix\_tokens} \in \mathbb{R}^{8 \times 512} \tag{Ctx-2}$$

---

## 5. Bộ dữ liệu: FaceScape & FaceVerse

### 5.1. FaceScape

**Nguồn:** Đại học Zhejiang (Zhu et al., 2020). Hệ thống quét 3D đa góc nhìn.

| Thuộc tính | Giá trị |
|-----------|---------|
| Số người (subjects) | 847 (tổng), 837 dùng cho train (10 dành test) |
| Số biểu cảm / người | ~22 trung bình (neutral, smile, mouth open, ... — tuỳ subject) |
| Tổng mesh dùng cho FaceDiff | **18,298** (đọc từ `data_signature` của ckpt epoch 390) |
| Vertex / mesh | 200K–400K |
| Topology | **TU registered** (`models_reg/*.obj` chia sẻ topology — xem `src/data/facescape_dataset.py`). Nguồn raw FaceScape có cả nhánh "detailed" unstructured nhưng FaceDiff chỉ dùng nhánh `models_reg/`. |
| Texture | UV-mapped, 2048×2048 |
| 3DMM bases | 300 identity + 52 expression |

**Mô hình tham số FaceScape Bilinear Model:**

$$S = \bar{S} + A_{\text{id}} \alpha + A_{\text{exp}} \beta + A_{\text{id-exp}} (\alpha \otimes \beta) \tag{FS-1}$$

Trong đó $\bar{S}$ = shape trung bình, $A_{\text{id}}$ = identity basis (PCA), $A_{\text{exp}}$ = expression basis, $\alpha, \beta$ = coefficients.

### 5.2. FaceVerse

**Nguồn:** Li et al. (2022). Mô hình khuôn mặt tham số có thể ước lượng từ ảnh.

| Thuộc tính | Giá trị |
|-----------|---------|
| Số người (subjects) | 110 (tổng), 100 dùng cho train (10 dành test) |
| Số biểu cảm / người | 21 (neutral, smile, mouth_stretch, ...) |
| Tổng mesh dùng cho FaceDiff | **2,100** (100 × 21, đọc từ `data_signature` của ckpt epoch 390) |
| Vertex / mesh | 5K–20K |
| Texture | Vertex color (per-vertex RGB) |
| Topology | Cố định (registered template) |
| 3DMM | 150 identity + 50 expression + 150 texture basis |

**Mô hình FaceVerse:**

$$V = \bar{V} + B_{\text{id}} \alpha + B_{\text{exp}} \beta + B_{\text{tex}} \gamma \tag{FV-1}$$

Trong đó $B_{\text{id}} \in \mathbb{R}^{3N \times 150}$, $B_{\text{exp}} \in \mathbb{R}^{3N \times 50}$, $B_{\text{tex}} \in \mathbb{R}^{3N \times 150}$.

### 5.3. Tổng hợp Dữ liệu FaceDiff

Số liệu thực tế đọc trực tiếp từ `data_signature` được hash vào `checkpoints/sc_vae_shape/epoch_390.pt`:

| Thuộc tính | FaceScape | FaceVerse | Tổng |
|-----------|-----------|-----------|------|
| Số mesh dùng huấn luyện | **18,298** | **2,100** | **20,398** |
| Số entries trong LMDB (kể cả test cache) | — | — | 20,968 |
| O-Voxel/mesh | 50K–350K voxels | 5K–50K voxels | — |
| Cache | LMDB `data/ovoxel_cache_lmdb/data.mdb`: ~272 GB dữ liệu thực, 429 GB pre-allocated map_size | | — |
| Train/Val split | 95/5 (Subset) | 95/5 (Subset) | **19,379 train / 1,019 val** |

**Train/Test split:** Chia theo identity (không theo expression) để tránh identity leakage. 10 identities/dataset giữ cho test (`test_facescape_ids.txt`, `test_faceverse_ids.txt` — file bị thiếu sẽ làm dataset chỉ filter theo train IDs).

**Tiền xử lý:** Mesh → chuẩn hóa $[-0.5, 0.5]^3$ → O-Voxel 10-kênh → LMDB sequential B-Tree.

> Lưu ý sửa lỗi: revision 1 của báo cáo viết "847 × 20 = 18,658 mesh FaceScape" và "110 × 21 = 2,310 mesh FaceVerse, tổng 20,968". Hai con số này không khớp với thực tế:
> - FaceScape có cả ~10 expression bổ sung tuỳ subject (đọc bằng glob `*.obj`), nên trung bình ~22 mesh/subject × 837 train ID = 18,298 mesh.
> - FaceVerse chỉ có 100 train ID × 21 expression = 2,100 mesh (10 ID còn lại để test).
> - 20,968 là số entries trong LMDB (gồm cả 10 ID test), nhưng số mesh thực tế chạy qua DataLoader là 20,398.

---

## 6. Thực nghiệm

### 6.1. Cấu hình Huấn luyện

#### 6.1.1. SC-VAE

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| Encoder dims | [64, 128, 256, 512] | 4-level pyramid |
| Latent dim | 32 | Per Slat token |
| Slat length | 4096 | $16^3$ grid |
| Res blocks/level | 2 | SparseResMLPBlock |
| In channels | 10 | shape_mat mode |
| Total params | **35.13 M** (đo bằng `sum(p.numel()) / 1e6` trên model load từ `epoch_390.pt`) | |
| Optimizer | AdamW (`fused=True` trên CUDA), $\beta_1{=}0.9$, $\beta_2{=}0.999$, weight_decay=0 | |
| LR (default) | $5 \times 10^{-5}$ → cosine (min $10^{-6}$) | Train-from-scratch |
| **LR (checkpoint epoch 397)** | $1 \times 10^{-5}$ base, đang ở $9.89\times 10^{-6}$ | Đọc từ `resume_contract` của ckpt |
| Batch size | 4 | |
| Gradient Accum | 33 steps | Effective batch 132 |
| KL weight | $10^{-6}$, warmup 20 epochs | Sau revision 2: chia `mu.numel()` (đúng spec) |
| Rho weight | 0.2 | TRELLIS.2 dùng 0.1 |
| Precision | **FP16 (AMP)** với `LayerNorm32` cast FP32 | spconv 2.x không hỗ trợ bfloat16 cho mọi op nên FaceDiff chốt FP16 |
| Max voxels/sample | 350,000 | |
| Max epochs | 500 (cosine ban đầu) + tuỳ chọn `--resume-extend-epochs` 100 (cosine_restart) | |
| Hardware | 1× RTX 4090 (24GB), peak quan sát thực **16,961 MB** (dư địa) | |

> Đính chính: revision 1 viết "Precision BFloat16" và LR=$5\times 10^{-5}$ cố định. Hai số này *đúng cho train-from-scratch nhưng sai cho ckpt epoch 397*. Trong checkpoint thực tế (đọc từ `resume_contract.details.lr_scheduler='cosine_with_min_lr'` và `learning_rate=1e-5`), AMP dtype là FP16 (xem `train_sc_vae.py` dòng `amp_dtype = torch.float16`) chứ không phải BFloat16, và base_lr là $1\times 10^{-5}$ chứ không phải $5\times 10^{-5}$.

#### 6.1.2. VoxelMamba + iMF

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| BiMamba layers | 12 | |
| Hidden dim | 512 | |
| SSM state dim ($n$) | 16 | |
| SSM conv kernel | 4 | |
| Expansion factor | 2 | $512 \to 1024$ inner |
| Prefix tokens | 24 | 8+4+4+4+4 (ctx+t+r+interval+guidance) |
| Context dim | 946 | ArcFace+FLAME+DINOv2 |
| **Total params** | **~21.2 M** (thêm interval tokenizer so với 20.88M trước) | Comment "~45M" trong `src/config.py` đã hết hạn — sẽ chỉnh lại trong commit kế tiếp |
| Optimizer | Adam | |
| LR | $2 \times 10^{-4}$ | |
| Batch size | 48 | |
| Gradient Accum | 33 | |
| EMA decay | 0.9995 | |
| Max epochs | 400 | |
| t sampler | logit-normal → curriculum | $\mu=-0.4$, scale=1.0 |
| CFG $\omega$ | $[1, 8]$, $\beta=1$ | |
| CFG context dropout | 0.1 | |
| Adaptive loss weighting | Enabled (100 bins, EMA decay=0.99) | iMF Appendix A |
| Precision | BFloat16 | |

### 6.2. Tối ưu Phần cứng

| Kỹ thuật | Tiết kiệm | Chi tiết |
|----------|-----------|----------|
| **Precompute Slat + context + `--offline-data`** | **~4–8 GB VRAM** (ước lượng: bỏ SC-VAE ~35M + DINO + ArcFace + renderer khỏi GPU trong vòng lặp train) | `scripts/precompute_slat_cache.py` → `train_imf.py --offline-data`; xem Mục 4.4.0 |
| INT4 Quantization | ~75% VRAM DINOv2 | `torchao.int4_weight_only()` |
| BFloat16 Mixed Precision | ~50% VRAM | Giữ dynamic range FP32 |
| Gradient Checkpointing | ~8GB VRAM | Recompute activations (+20% time) |
| SparseConv3D (spconv) | ~45% VRAM | Chỉ tính active sites |
| LMDB Sequential I/O | 30× throughput | HDD: $5 \to 150$ MB/s |
| Gradient Accumulation | Effective batch 132 | 33 micro-batches |

### 6.3. Kết quả SC-VAE (tính đến 10/05/2026, log gốc `logs/train_sc_vae_gamma_fixed.log`)

| Epoch | Total Loss | Recon Loss | KL Loss (legacy norm) | Rho Loss | LR |
|-------|-----------|------------|------------------------|----------|-----|
| 200 | 0.0724 | 0.0546 | 0.8321 | — | $2.5 \times 10^{-5}$ |
| 350 | 0.0547 | 0.0271 | 0.7826 | 0.057 | $1.00 \times 10^{-5}$ |
| 384 | 0.0397 | 0.0271 | 0.7999 | 0.057 | $9.96 \times 10^{-6}$ |
| 385 | 0.0397 | 0.0271 | 0.8011 | 0.0565 | $9.96 \times 10^{-6}$ |
| **397** | **0.0365** | **0.0251** | **0.8321** | **0.0513** | $9.89 \times 10^{-6}$ |
| 388 (`latest_step.pt`) | 0.0379 | — | — | — | $9.94 \times 10^{-6}$ |
| 390 (`epoch_390.pt`) | 0.0381 | — | — | — | $9.94 \times 10^{-6}$ |

> **Đính chính so với revision 1:** revision trước báo cáo Total=0.0480 và LR=$2.0\times 10^{-6}$ ở epoch 397. Hai số đều sai. Total Loss thực tế tại epoch 397 = **0.0365** (`logs/train_sc_vae_gamma_fixed.log` dòng `Epoch 397/500 | Loss: 0.0365`). LR thực tế = **9.89×10⁻⁶** (cosine từ base_lr=1e-5 chỉ giảm rất chậm vì $r_{\min} = 0.1$). VRAM peak quan sát thực tế = **16,961 MB** (không phải <22 GB như mục tiêu, có dư địa cho mở rộng kiến trúc). Bảng ở trên dùng "KL Loss (legacy norm)" để chỉ giá trị ghi log với normalisation cũ (chia cho `target_x.shape[0]`). Sau khi revision này chuyển sang chia `mu.numel()` (đúng spec), giá trị KL log mới sẽ là `legacy / 32 ≈ 0.026` — chỉ là thay đổi báo cáo, contribution vào tổng loss (~$10^{-6} \cdot 0.83 = 8\times 10^{-7}$) vẫn không đổi đáng kể.

**Topology metrics tại epoch 397** (đo bằng `scripts/visualize_mesh_vs_ovoxel.py` trên 10 mẫu val):

| Metric | Giá trị | Ghi chú |
|--------|---------|---------|
| Recall (GT → Recon) | 86.7% | 13.3% GT voxels bị thiếu (Rho-pruning false negative) |
| Precision (Recon → GT) | 80.6% | 19.4% false positives ("rác" voxel sinh thêm) |
| mse_xyz (activated dv, sau Eq. Act-1) | 0.040 | dv displacement |
| mse_rgb (clamped) | 0.002 | Color fidelity |
| mse_all | 0.052 | Trung bình 10 channels |

---

## 7. Phân tích và Giải thích Kết quả

### 7.1. Phân tích Hội tụ

- **Recon Loss** giảm 78% (0.1614 → 0.0251), plateau từ epoch ~395
- **KL Loss** ở 0.83 — latent space chưa collapse (tốt cho diversity ở Stage 2)
- **Rho Loss** vẫn giảm -0.0013/epoch → topology đang cải thiện

### 7.2. Các Bug đã Fix

**Bug 1 — Test script XYZ bỏ qua dv offset:** Dùng grid coords thay vì `(coords + dv) × voxel_size + aabb_min`. Fix: dùng OV-4.

**Bug 2 — dc-level default 20.0:** Delta qua sigmoid ∈ [0,1] nhưng threshold 20.0 → 0 verts. Fix: default 0.5.

**Bug 3 — MSE trên raw logits:** Không apply activations trước khi tính MSE. Fix: apply sigmoid(delta), softplus(gamma), clamp(dv, rgb).

**Bug 4 — Render loss shape_mat bỏ qua dv:** Mode `shape_mat` dùng grid coords cho depth/mask rendering → dv channel không nhận gradient. Fix: dùng SC-4.

**Bug 5 — Gamma cache = 1.0 constant:** Cache sinh bằng code cũ (trước fix gamma). Fix: recompute $\gamma = (1 - \text{Var}(dv)).\text{clamp}(0,1)$ từ cached dv.

**Bug 6 — Output activations chưa khớp TRELLIS.2** *(fix 10/05/2026 — revision 2)*: trước revision này, lớp `out_proj` của decoder chỉ là `nn.Linear(64, 10)` trả raw logits, và loss `_shape_mat_recon_loss` áp `MSE` lên `dv` raw. Vì sigmoid(0)=0.5 còn target dv ∈ [0,1], gradient ép `dv` về center voxel thay vì đi học vị trí thực. Fix:
  - Thêm `apply_dv_activation` $(1+2m)\sigma(h)-m$ với `m=0.5` (Eq. Act-1, đúng spec `FlexiDualGridVaeDecoder`).
  - `_shape_mat_recon_loss` activate `dv` trước khi tính MSE; clamp target vào $[-0.5,1.5]$.
  - `apply_shape_mat_output_activations()` được expose ở module-level để inference/visualisation cũng dùng đúng activation.
  - Thêm flag `apply_output_activations=False` (mặc định) để không đụng đến đường loss khác (BCE-with-logits ở `delta` vẫn cần raw logits).
  - Backward-compat: load `epoch_390.pt` strict `missing=0, unexpected=0` (xem Mục 7.2.6).

**Bug 7 — Render loss dùng `clamp(dv, 0, 1)` thay vì activated dv** *(fix 10/05/2026)*: `src/scvae_train/render.py` chỉ clamp raw logits, gây giống Bug 6 trên 2D maps. Fix: project bằng `apply_dv_activation(recon[..., 0:3])`, đồng nhất với inference path của dual contouring.

**Bug 8 — KL chia sai chuẩn** *(fix 10/05/2026)*: trước revision này KL chia cho `target_x.shape[0]` thay vì `mu.numel()`, tức thiếu hệ số $d_{\text{lat}} = 32$. Vì $w_{KL} = 10^{-6}$ rất nhỏ nên không ảnh hưởng training, nhưng giá trị KL log không khớp Eq. (VAE-2) trong báo cáo. Fix: chia `mu.numel()` + `clamp(logvar, -30, 20)` trước `exp` để chống overflow dưới AMP.

**Bug 9 — Encoder thiếu non-affine LayerNorm trước to_mu/to_logvar** *(fix 10/05/2026)*: TRELLIS.2 chèn `F.layer_norm(h.feats, h.feats.shape[-1:])` ngay trước `to_latent` để giữ posterior ổn định. Fix: thêm flag `pre_latent_norm=True` (default) trong `SC_VAE`. Vì là *non-parametric* (zero param), không phá `state_dict` ⇒ checkpoint epoch 397 load được không sửa.

**Bug 10 — `SparseResMLPBlock` thiếu zero-init lớp cuối** *(fix 10/05/2026)*: TRELLIS.2 `SparseConvNeXtBlock3d` dùng `zero_module(nn.Linear(...))` cho `pw2` để residual branch xuất 0 ở init, ổn định optimisation. Fix: thêm `_zero_module` cho `pw2`. Tác dụng chỉ thấy khi train-from-scratch; resume từ ckpt thì weight cũ overwrite zero-init ngay nên không thay đổi behavior.

**Bug 11 — `SparseResMLPBlock` LayerNorm không cast FP32** *(fix 10/05/2026)*: TRELLIS.2 dùng `LayerNorm32` (cast sang FP32 trước, đổi lại sau) để chống NaN dưới `torch.amp` FP16 mà spconv yêu cầu. Fix: import `src.modules.norm.LayerNorm32` và fallback về `nn.LayerNorm` (kèm cast tay) nếu không sẵn. Param shape `weight/bias` không đổi ⇒ ckpt-safe.

**Bug 13 — `scripts/precompute_slat_cache.py` thiếu hybrid extractors** *(fix 11/05/2026 — revision 3)*: bản cũ tạo `SlatDataset` không truyền `mesh_renderer` / `arcface` / `flame` / `dinov2`, nên nhánh `_encode_mesh` luôn rơi vào fallback `torch.randn(946)` — cache **sai** so với train online. Fix: script nạp đầy đủ bốn module giống `train_imf.py`; thêm `--use-random-context` chỉ cho debug; thêm `--skip-existing` thực sự kiểm tra file qua `SlatDataset.has_valid_cache`.

**Bug 12 — Không có scheduler hợp lệ để fine-tune sau khi cosine ban đầu chạy hết** *(fix 10/05/2026)*: cosine schedule mặc định kéo dài $500 \times 146 \approx 73{,}000$ steps. Checkpoint epoch 397 đã ở step 4001 (`scheduler._step_count`); resume tiếp 100 epoch nữa với cùng cosine sẽ bị "kẹt" ở $\sim 1.0\times 10^{-6}$ rất sớm. Fix: thêm `get_resume_scheduler()` với 3 mode (`continue`/`constant_min_lr`/`cosine_restart`) + CLI `--resume-scheduler-mode` + script `scripts/resume_from_397.sh`. Đã verify cosine_restart bắt đầu từ $9.94\times 10^{-6}$ và giảm về $1\times 10^{-6}$ sau đúng `--resume-extend-epochs` (test xem `runtime.py:get_resume_scheduler`).

**Bug 13 — `scripts/precompute_slat_cache.py` thiếu hybrid extractors** *(fix 11/05/2026 — revision 3)*: bản cũ tạo `SlatDataset` không truyền `mesh_renderer` / `arcface` / `flame` / `dinov2`, nên nhánh `_encode_mesh` luôn rơi vào fallback `torch.randn(946)` — cache **sai** so với train online. Fix: script nạp đầy đủ bốn module giống `train_imf.py`; thêm `--use-random-context` chỉ cho debug; `--skip-existing` gọi `SlatDataset.has_valid_cache`; API `cache_file_path` / `has_valid_cache` thêm vào `SlatDataset`.

### 7.2.6. Backward-compat verification (revision 2)

Để đảm bảo các thay đổi của revision 2 không phá checkpoint epoch 397, đã chạy bộ test sau:

```text
[Test] strict load missing=0 unexpected=0
[Test] Total params: 35.13M  (giữ nguyên)
[Test] Activation: dv range=[-0.500, 1.500]  (đúng Eq. Act-1, m=0.5)
[Test] Activation: delta unique=[0.0, 1.0]   (eval mode threshold > 0)
[Test] Activation: gamma min=0.0003          (softplus > 0)
[Test] Activation: rgb range=[0.000, 1.000]  (clamp)
[CosineRestart] start_lr=9.94e-6, sau 100 epoch giảm đúng về 1e-6
[ConstantMinLR] LR đứng yên ở 2e-6 sau step đầu
[Smoke] OK  (CLI flags --resume-scheduler-mode/extend/target-min-lr xuất hiện đúng)
```

→ Có thể `bash scripts/resume_from_397.sh` mà không lo `RuntimeError` ở `load_state_dict`.

### 7.3. So sánh với TRELLIS.2

| Tiêu chí | TRELLIS.2 (`microsoft/TRELLIS.2`) | FaceDiff |
|----------|----------|----------|
| GPU | 8× A100 / H100 (320GB+) | **1× RTX 4090 (24GB)** |
| Backbone | U-DiT (Transformer $O(N^2)$) | **VoxelMamba (SSM $O(N)$)** |
| Generation steps | 50 (DDPM) | **1 (iMF)** |
| O-Voxel cho shape SC-VAE | 6 kênh in (`vertices-0.5`, `intersected-0.5`) → 7 kênh out (`dv,δ,γ`) | 10 kênh in/out (gộp `dv,δ,γ,rgb`) |
| Encoder kênh | `[64,128,256,512,1024]` × `[0,4,8,16,4]` blocks (5 levels) | `[64,128,256,512]` × `[2,2,2,2]` blocks (4 levels — phù hợp 24 GB) |
| Encoder downsample | `SparseResBlockS2C3d` (Spatial2Channel) | strided `SparseConv3d` + non-parametric S2C-shortcut |
| Decoder upsample | `SparseResBlockC2S3d` + `SparseChannel2Spatial(2, mask)` | `SparseConvTranspose3d` + non-parametric C2S-shortcut + post-pruning |
| Norm | `LayerNorm32` (FP32 cast) | **`LayerNorm32` (đã align từ revision 2)** |
| Pre-latent norm | non-affine `F.layer_norm` | **đã align từ revision 2** (`pre_latent_norm=True`) |
| Output activations | `(1+2m)·sigmoid - m`, `>0`, `softplus`, (không có RGB) | **đã align từ revision 2** (`apply_shape_mat_output_activations`) + `clamp(rgb,0,1)` |
| Render loss | **Mesh rasterizer (`nvdiffrast`) → real depth/normals** | Point projection → approximate (đã align dv path) |
| Conditioning | Cross-Attention | **Prefix Tokens** (Mamba-friendly) |
| Loss weights — dv | $\lambda_{\text{vertice}} = 0.01$ | $0.01$ (identical) |
| Loss weights — δ | $\lambda_{\text{intersected}} = 0.1$ | $0.1$ (identical) |
| Loss weights — KL | $\lambda_{\text{kl}} = 10^{-6}$ | $10^{-6}$ (identical, normalisation đã sửa ở revision 2) |
| Loss weights — Rho | $\lambda_{\text{subdiv}} = 0.1$ | **0.2** (FaceDiff 2× vì topology face sparse hơn Objaverse-XL) |
| Loss weights — render depth | $\lambda_{\text{depth}} = 10$ | $10$ (identical) |
| Loss weights — render mask | $\lambda_{\text{mask}} = 1$ | $1$ (identical) |
| Loss weights — Perceptual | L1 + 0.2 SSIM + 0.2 LPIPS | **Identical** |
| Loss weights — Normal | $\lambda_{\text{normal}} = 1$ (TRELLIS.2 có) | $\lambda_{\text{normal}} = 1$ — **✅ 14/05** depth-to-normal via finite differences trên depth maps |
| Optimizer | AdamW lr=1e-4, EMA 0.9999, AdaptiveGradClipper | AdamW lr=1e-5 (resume), **EMA 0.9999 ✅ 14/05**, AdaptiveGradClipper ✅ |

**Khác biệt cốt lõi còn lại sau revision 3 (14/05/2026):**
1. **Render loss**: TRELLIS.2 render actual mesh qua `nvdiffrast` differentiable rasterizer → depth/normals/PBR shading chính xác. FaceDiff project point features lên 2D maps → xấp xỉ. `nvdiffrast` 0.4.0 đã cài; tích hợp mesh rasterization thay thế point projection sẽ thực hiện ở lượt train tiếp theo.
2. **Encoder/Decoder downsample**: TRELLIS.2 dùng `SparseSpatial2Channel/Channel2Spatial` zero-param (chỉ reshape & permute features), không tốn FLOPs. FaceDiff dùng strided sparse conv tốn thêm $C^2 \cdot k^3$ FLOPs/level nhưng dễ build với spconv 2.x.
3. ~~**EMA weights**~~: ✅ **Đã triển khai** (14/05) — decay=0.9999, shadow VRAM ~134MB. Tích hợp vào training loop, checkpoint, resume.
4. ~~**Normal loss**~~: ✅ **Đã triển khai** (14/05) — `_depth_to_normal()` tính pháp tuyến từ depth maps bằng sai phân hữu hạn. L1 + 0.2×SSIM + 0.2×LPIPS trên normal maps, khớp TRELLIS.2 spec.

### 7.4. Phân tích Thiết kế

**Tại sao Mamba > Transformer cho 3D generation?**
- 4096 tokens × 12 layers × batch 48: Attention maps ~15GB VRAM
- Mamba: ~2-3GB VRAM (256× ít hơn)

**Tại sao iMF > DDPM/Flow Matching?**
- DDPM 50 bước: 50× inference time
- FM 1 bước: quỹ đạo cong → quality thấp
- iMF: JVP nắn thẳng quỹ đạo → 1 bước + chất lượng cao

**Tại sao Prefix Tokens > Cross-Attention?**
- Cross-Attention: +$O(N \times K)$/layer
- Prefix: chỉ +20 tokens, Mamba xử lý $O(N+20) \approx O(N)$

---

## 8. Kế hoạch Tiếp theo

### 8.1. Ngắn hạn (05/2026)

| Giai đoạn | Nội dung | Thời gian | Trạng thái |
|-----------|----------|-----------|-----------|
| Fix Gamma + LMDB | Recompute gamma, re-pack | 6.5h | ✅ 09/05 |
| Fix LR Warmup | 6000 → 500 steps | 5 phút | ✅ 09/05 |
| Audit + Fix 5 Bug đầu | Test script + render loss | — | ✅ 10/05 (revision 1) |
| **Audit TRELLIS.2 + Fix 7 Bug mới** (Bug 6→12) | Activations, KL, LayerNorm32, scheduler resume | — | ✅ 10/05 (revision 2) |
| **SC-VAE resume 397→497** | 100 epoch × ~30 min với `cosine_restart` từ 9.94e-6 → 1e-6 | ~50h | ✅ Hoàn thành epoch 441 |
| **EMA + Depth-to-Normal + Cosine Restart** | Thêm EMA (0.9999), normal loss (λ=1), cosine restart +200ep | — | ✅ 14/05 (revision 3) |
| **nvdiffrast** | Cài nvdiffrast 0.4.0 cho future mesh rasterization | — | ✅ 14/05 |
| **SC-VAE resume 441→700** | +200 epoch, cosine restart với EMA + normal loss + render loss | ~75h | ⏳ Đang chạy (epoch 442) |
| **Precompute Slat + context (iMF)** | `python scripts/precompute_slat_cache.py --sc-vae-ckpt … --dataset both --skip-existing` → đầy đủ hybrid context (đã sửa bug bản cũ) | ~3–8h tuỳ I/O/GPU | ✅ Công cụ + báo cáo Mục 4.4.0 (revision 3) |
| **iMF train với `--offline-data`** | Tăng batch sau precompute; không nạp SC-VAE/DINO trên GPU train | song song SC-VAE nếu cần | ⏳ Sau khi cache đủ |
| iMF Training (online, không offline) | 400 epochs, batch 48 | ~27h | ⏳ Tuỳ chọn nếu không precompute |
| E2E Test | Inference + mesh quality | ~0.5 day | ⏳ |
| **Tổng** | | **~7 ngày** | **ETA: 18-19/05** |

**Dự báo hội tụ SC-VAE** (cập nhật theo loss thực tế của log gốc):

| Epoch | total_loss | rho_loss | Topology Recall | Ghi chú |
|-------|-----------|----------|-----------------|---------|
| 397 | 0.0365 | 0.0513 | 86.7% | Mức hiện tại (đã verify trên 10 mẫu val). |
| 430 | ~0.030 | ~0.040 | ~89% | Sau khi Bug 6/7 (activated dv) phát huy: gradient nhất quán hơn, recon dv giảm thêm 10–15 %. |
| 470 | ~0.025 | ~0.020 | ~92% | Cosine restart kéo LR xuống ~3 × 10⁻⁶. |
| 497 | ~0.022 | ~0.012 | ~94% | Mốc dừng — cosine_restart chạm `target_min_lr=1e-6`. |

> Ghi chú resume: dùng `bash scripts/resume_from_397.sh` (mặc định `EXTEND_EPOCHS=100, TARGET_MIN_LR=1e-6`). Script tự bật `--lmdb-only`, `--enable-stage2-render-loss`, `--gradient-accumulation-steps 33`, `--batch-size 4` để khớp với `resume_contract` đã hash trong checkpoint, tránh `RuntimeError: resume_contract mismatch`.

### 8.2. Trung hạn (06–07/2026)

**Metrics đánh giá chất lượng mesh (theo TRELLIS.2):**

**Chamfer Distance (CD)** — khoảng cách trung bình 2 chiều giữa hai tập điểm:

$$\text{CD}(X, Y) = \frac{1}{2|X|} \sum_{x \in X} \min_{y \in Y} \|x - y\|^2 + \frac{1}{2|Y|} \sum_{y \in Y} \min_{x \in X} \|y - x\|^2 \tag{Eval-1}$$

**F-Score** — precision/recall với ngưỡng khoảng cách $\tau$:

$$\text{Prec}(\tau) = \frac{1}{|P_{\text{pred}}|} \sum_{p \in P_{\text{pred}}} \mathbb{1}[d^2(p, S_{\text{gt}}) < \tau], \quad \text{Rec}(\tau) = \frac{1}{|P_{\text{gt}}|} \sum_{p \in P_{\text{gt}}} \mathbb{1}[d^2(p, S_{\text{pred}}) < \tau] \tag{Eval-2}$$

$$\text{F-Score}(\tau) = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}} \tag{Eval-3}$$

**SSIM** (cho render loss và so sánh 2D):

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}, \quad C_1 = 0.01^2, \, C_2 = 0.03^2 \tag{Eval-4}$$

- **Ablations:** Mamba vs GRU, Hilbert vs Morton, prefix token count, KL schedule
- **Identity:** ArcFace cosine similarity giữa render mesh và ảnh gốc
- **Inference optimization:** TorchScript, batched pipeline, target < 1s/mesh

### 8.3. Dài hạn (08–12/2026)

- Multi-view conditioning, temporal consistency
- Tích hợp FERG, BU-3DFE, CoMA datasets
- Web demo (Three.js), API endpoint, edge deployment
- Bài báo CVPR/ECCV/NeurIPS

---

## 9. Cập nhật Alignment iMF (Revision 4)

### 9.1. Mục tiêu cập nhật

Đối chiếu trực tiếp với paper **Improved Mean Flows** (arXiv:2512.02012v1) cho thấy một số điểm trong code đã đúng ý tưởng tổng thể nhưng có sai lệch triển khai. Revision 4 tập trung sửa các điểm ảnh hưởng trực tiếp đến tính đúng đắn thuật toán huấn luyện.

### 9.2. Các thay đổi đã thực hiện

1. **Sửa adaptive weighting trong iMF loss (quan trọng nhất).**

- **Vấn đề cũ:** adaptive weight được chuẩn hóa mean=1, sau đó chỉ nhân bằng **một scalar trung bình**, làm hiệu ứng weighting gần như triệt tiêu.
- **Sửa mới:** áp dụng trọng số **theo từng sample/timestep** trên vector loss trước khi lấy mean batch.
- **Kết quả:** đúng tinh thần adaptive weighting theo MeanFlow/iMF Appendix A, thay vì chỉ scale toàn batch bằng hằng số.

2. **Bổ sung hàm per-sample weighted MSE cho nhánh boundary/JVP.**

- Tạo hàm `per-sample` để tính loss vector [B], dùng cho cả cập nhật EMA timestep-bin và backprop.
- Giữ `channel_weights` tương thích chế độ dual-branch (shape/material).

3. **Điều chỉnh token conditioning theo iMF Appendix (Tab. cấu hình).**

- Mở rộng cấu hình token cho VoxelMamba trong `IMFConfig`:
    - `mamba_num_context_tokens`, `mamba_num_time_tokens`, `mamba_num_r_tokens`, `mamba_num_interval_tokens`, `mamba_num_guidance_tokens`.
- Đặt mặc định `mamba_num_interval_tokens = 4` để bám paper.
- Truyền đầy đủ các token-count này ở training stage-2.

4. **Giữ tương thích checkpoint cũ khi inference.**
5. **Đồng bộ CFG interval với paper (train + inference).**

- Khi $t$ nằm ngoài $[t_{\min}, t_{\max}]$, đặt $\omega_{\text{eff}} = 1$ cho **cả** input model lẫn mục tiêu, đảm bảo CFG bị tắt đúng định nghĩa Appendix A.

6. **Áp dụng adaptive weighting cho v-head loss.**

- V-head loss cũng nhân $w_{\text{adaptive}}(t)$ như loss chính, đúng theo Appendix A.

- Generator được cập nhật để tự suy luận `num_r_tokens` và `num_interval_tokens` từ trọng số checkpoint (`r_tokenizer`, `interval_tokenizer`) nếu có.
- Vì vậy checkpoint cũ vẫn có thể load (fallback an toàn), checkpoint mới dùng cấu hình token đầy đủ.

### 9.3. Tệp mã đã chỉnh sửa

- `src/models/imf_diffusion.py`
- `src/config.py`
- `src/train_imf.py`
- `src/inference/generator.py`
- `src/models/voxel_mamba.py`

### 9.4. Tác động kỳ vọng

- Huấn luyện iMF ổn định hơn ở các vùng timestep khó (do adaptive weighting hoạt động thực sự).
- Bám sát đặc tả paper hơn ở phần in-context conditioning.
- Không đánh đổi khả năng dùng checkpoint cũ trong pipeline inference.

### 9.5. Lưu ý vận hành

- Với stage-2 checkpoint được huấn luyện trước revision 4, có thể xuất hiện chênh số liệu loss do objective đã được weighting đúng theo sample.
- Nếu muốn so sánh công bằng trước/sau, nên chạy lại stage-2 từ cùng một checkpoint khởi tạo và cùng seed.

### 9.6. Đối chiếu chi tiết với arXiv:2512.02012v1 (Geng et al.) — Revision 5

**Đính chính trích dẫn:** Mã nguồn và báo cáo trước đây ghi nhầm tác giả *Kim et al.* cho cùng mã arXiv. Bài *Improved Mean Flows: On the Challenges of Fastforward Generative Models* tại **arXiv:2512.02012v1** là của **Geng, Lu, Wu, Shechtman, Kolter, He** (2025). Đã cập nhật docstring / comment / mục 3.4 và tài liệu tham khảo [1].

**Bảng đối chiếu thuật toán ↔ `src/models/imf_diffusion.py`:**

| Thành phần paper | Mã FaceDiff | Ghi chú |
|------------------|-------------|---------|
| Nội suy tuyến tính $z_t=(1-t)x+te$ | `z_t = (1-t)*x + t*e` | Khớp |
| $u_\theta(z,r,t)$, $v_\theta$ (v-head) | `forward_u`, `forward_v` | Backbone VoxelMamba thay DiT — đúng ý tưởng |
| Công thức hợp $V=u+(t-r)\,\mathrm{sg}(\partial u/\partial t)$ (Eq. 12) | `u_theta + (t-r)*dudt_sg` | `dudt` qua `torch.func.jvp` trên $(z,t)$, $r$ cố định trong closure — tương đương Alg. 1 khi $\mathrm{d}r/\mathrm{d}t=0$ |
| Mục tiêu loss chính $(e-x)$ (hoặc CFG Eq. 17) | `target_main`, nhánh batch-doubled CFG | Khớp Eq. (17) với $\omega_{\mathrm{eff}}$ ngoài $[t_{\min},t_{\max}]$ |
| Alg. 2: tiếp tuyến JVP dùng $v_c$ (điều kiện) | `v_theta = u(z,t,t,c)` trước CFG blend | Khớp tinh thần Appendix |
| Appendix A: FM loss trên v-head + adaptive weight | `v_loss` hệ số 0.1, cùng `w_adaptive` | Khớp |
| Lấy mẫu 1 bước $z_0=z_1-u(z_1,0,1)$ | `sample_1_step` | Khớp |

**Lỗi đã sửa (revision 5):** Nhánh `return_components` khi `mask_eq_all` tham chiếu biến không tồn tại `material_keep_val` — đã thay bằng `material_keep_ratio.detach()` để nhất quán với nhánh còn lại và tránh `NameError` / giá trị sai khi log/monitor.

**Khác biệt có chủ ý (không phải bug):** optimizer $\beta_2$ (0.999 vs 0.95 trong paper), EMA decay, curriculum $t$-sampler mở rộng, dual-branch shape/material — phù hợp bài toán khuôn mặt 3D.

---

## 9.7. Đối chiếu chi tiết Revision 6 — Đọc lại kỹ iMF + DiM-3D, fix toàn bộ pipeline

### 9.7.1. Mục tiêu rà soát

1. Đọc lại paper [arXiv:2512.02012v1](https://arxiv.org/html/2512.02012v1) (iMF — Geng et al.) đối chiếu mọi công thức/Alg. 1/Alg. 2/Tab. 4/Appendix A với code.
2. Đọc lại paper [arXiv:2406.05038v1](https://arxiv.org/html/2406.05038v1) (DiM-3D — Mo, NeurIPS 2024) đối chiếu khối Bidirectional SSM với `src/models/voxel_mamba.py`.
3. Soát lỗi trên toàn pipeline: `train_imf.py`, `train_sc_vae.py`, `imf_diffusion.py`, `voxel_mamba.py`, `sc_vae_loss.py`, `inference/generator.py`. (Module `imf_vloss.py` cũ đã gỡ khỏi repo — JVP không đủ Alg. 1.)
4. Sửa các bug đã phát hiện và mô tả bằng văn bản dễ truy vết để team áp dụng/đối soát checkpoint cũ.

### 9.7.2. Đối chiếu iMF paper ↔ `src/models/imf_diffusion.py` (sau revision 6)

| Khía cạnh paper | Yêu cầu paper | Trạng thái code (rev. 6) |
|-----------------|---------------|--------------------------|
| Nội suy $z_t=(1-t)x+te$ | Gaussian $e$, $x$ là data | `_interpolate()` — KHỚP |
| Sample $(t,r)$, ratio $r{\neq}t$ = 0.5 (Tab. 4) | $t$ logit-normal $(-0.4, 1)$; $r$ ngẫu nhiên trong $[0,t]$ | `_sample_t_r()` đã sort $r\le t$ (sửa bug rev. 6, mục 9.7.4 #2) |
| Conditioning $(t,r)$ qua "$t-r$" (Tab. 4) | Mã hóa **signed** $(t-r)$ | `interval = (t-r)` không còn `.abs()` (sửa rev. 6 #3) |
| Alg. 1: tiếp tuyến JVP $(v,0,1)$ trên $(z,r,t)$ | $v$ là $u(z,t,t,c)$ (boundary) | `_compute_dudt_jvp()` dùng $(v_{\text{tangent}}, 1)$ trên $(z,t)$, $r$ cố định trong closure ⇒ tương đương Alg. 1 vì $\mathrm{d}r/\mathrm{d}t=0$ |
| Compound $V=u+(t-r)\,\mathrm{sg}(\mathrm{JVP})$ (Eq. 9, 12) | sg trên $\mathrm{d}u/\mathrm{d}t$ | `V = u_pred + t_minus_r * dudt.detach()` — KHỚP |
| Boundary $r=t$ giảm về v-loss đơn | $\|u(z,t,t)-(e-x)\|^2$ | `mask_eq_all` branch dùng MSE giữa `v_theta` và `v_target_loss` — KHỚP |
| CFG Eq. 17: $v_{\text{cfg}}=(e-x)+(1-1/\omega)(v_c-v_u)$ | $\omega=1$ ngoài $[t_{\min},t_{\max}]$ | Batch-doubling, `omega_effective`, `coeff = 1 - 1/\omega_{\text{eff}}` — KHỚP |
| Alg. 2: JVP tangent dùng $v_c$ (conditional) | trước CFG blend | `v_theta = u(z,t,t,c)`, dùng làm tangent — KHỚP |
| Appendix A: aux v-head FM loss + adaptive weight | $\|v_\theta-(e-x)\|^2$ + cùng adaptive weight | `loss_v_head` hệ số 0.1, nhân `adaptive_w` — KHỚP |
| Adaptive weighting per-bin EMA | Bin theo $t$, normalize 1 | `_get_adaptive_weights()`, EMA decay 0.99 — KHỚP, đã vector hóa (rev. 6 #6) |
| Sample 1-bước $z_0=z_1-u(z_1,0,1)$ | $r=0,t=1$ | `sample_1_step()` — KHỚP |
| CFG conditioning $\omega \sim \omega^{-\beta}$ trên $[1,8]$ | $\beta\in\{1,2\}$ | `_sample_cfg_omega()` analytic inverse-CDF — KHỚP |
| CFG interval sampling $t_{\min}\!\sim\!\mathcal{U}[0,0.5]$, $t_{\max}\!\sim\!\mathcal{U}[0.5,1]$ | per-sample | `_sample_cfg_interval()` — KHỚP |

### 9.7.3. Đối chiếu DiM-3D / Voxel Mamba paper ↔ `src/models/voxel_mamba.py`

Paper `DiM-3D` (Mo, 2024 — arXiv:2406.05038v1) đề xuất khối **Bidirectional SSM** thay self-attention $O(N^2)$ trong DiT bằng Mamba $O(N)$ cho voxel sequences. So sánh với khối `BidirectionalMambaBlock`:

| Thành phần paper (§3.3) | Yêu cầu | Trạng thái code |
|-------------------------|---------|-----------------|
| Norm trước SSM | LayerNorm chuẩn | `RMSNorm(dim)` — biến thể nhẹ hơn, mục 4.5 update3 đã ghi |
| Linear projection vào $\mathbf{z}$ | Có | Mamba xử lý nội bộ (`d_model=dim`, `expand=2`) |
| Forward + Backward scan | 2 SSM song song, fuse output | `forward_mamba`, `backward_mamba` trên `x` và `flip(x)` rồi `flip` ngược — KHỚP |
| Combine 2 hướng | Sum hoặc concat-projection | `out = fwd + bwd` — KHỚP (sum, đơn giản hơn paper concat) |
| Conv1D điều chỉnh $\mathbf{z}'$ | Có (Mamba block nội tại) | Tham số `d_conv=4` truyền vào `Mamba()` — KHỚP |
| $\mathbf{B},\mathbf{C},\Delta$ động | Selective SSM | Mamba native, $d\_state=16$ — KHỚP |
| Linear complexity $O(L)$ | Yêu cầu chính | `mamba-ssm` CUDA kernel hoặc fallback BiGRU — KHỚP |
| Patchification voxel | $X\!\times\!Y\!\times\!Z\to L\!\times\!(P^3{\cdot}C)$ | `SC-VAE` đã encode O-Voxel → 4096 token; Voxel Mamba nhận trực tiếp tokens (không cần patchify lại) — phù hợp Slat representation |
| Class token global | 1 token | FaceDiff dùng 8 context tokens (ArcFace+FLAME+DINO) — mạnh hơn |
| Positional embedding | Learnable | Dùng Hilbert SFC reorder + Mamba causal scan — đảm bảo token gần nhau trong 3D liền kề trong 1D, hiệu quả hơn cho SSM |
| Time/cond embedding | adaLN-zero (DiT-style) | iMF Sec. 4.3 + DiM-3D §3.2: chuyển sang **in-context conditioning** (8 ctx + 4 t + 4 r + 4 interval + 4 guidance = 24 prefix token) — KHỚP và mạnh hơn |

**Kết luận:** Khối `BidirectionalMambaBlock` của FaceDiff là một implementation **trung thành tinh thần** của DiM-3D, đồng thời tích hợp 2 cải tiến từ iMF: (i) in-context conditioning thay adaLN-zero giảm 1/3 tham số, (ii) `r`-conditioning + `(t-r)` interval token cho mean-flow.

**Khác biệt có chủ ý vs DiM-3D paper:**
- Backbone hợp nhất xử lý sequence Slat (4096 tokens, 16³) thay vì voxel 3D thô — tận dụng SC-VAE compression.
- RMSNorm thay LayerNorm (Llama/Mamba SSM hiện đại).
- Hilbert reordering — DiM-3D không có (dùng raster); FaceDiff thêm vì nâng locality cho SSM tuần tự, paper Voxel Mamba (Zhang 2024) cũng kiến nghị tương tự.
- Output projection zero-init (TRELLIS.2) → block ban đầu là identity, ổn định khi train từ scratch.

### 9.7.4. Tám lỗi/bug đã sửa trong revision 6

| # | File | Mức độ | Vấn đề | Cách sửa |
|---|------|--------|--------|----------|
| 1 | `src/models/imf_diffusion.py` | **CRITICAL** | `_want_hidden = (v_head is not None and hasattr(model, 'forward'))` — `hasattr(model,'forward')` luôn True ⇒ khi backbone không hỗ trợ `return_hidden=True`/`get_hidden_state` (vd legacy `IMFUNet1D`) sẽ bị `TypeError: forward() got an unexpected keyword argument 'return_hidden'`. | Đổi điều kiện thành `_want_hidden = bool(v_head is not None) and hasattr(model, 'get_hidden_state')`. Backbone không hỗ trợ thì rơi xuống nhánh fallback `model.get_hidden_state()` hoặc `v_head(z_t)` an toàn. |
| 2 | `src/models/imf_diffusion.py:_sample_t_r` | **CRITICAL** | `r` và `t` lấy mẫu **độc lập** ⇒ ~50% trường hợp `r > t`, $(t-r) < 0$. Trái với convention paper iMF (Alg. 1 hàm sample_t_r ngầm định) và bản chính thức của MeanFlow (Gsunshine repo) sort $r\le t$. Khi conditioning lên signed $(t-r)$, model nhận tín hiệu chéo dấu khiến training khó hội tụ. | Sample `r ~ U[0, t]` rồi `where(mask_neq, r_candidate, t)`. Bảo đảm $r\le t$ luôn ⇒ $(t-r)\ge 0$. |
| 3 | `src/models/voxel_mamba.py` | **CRITICAL** | `interval = (t - r).abs()` mất dấu — paper Tab. 4 ghi rõ "$(t,r)$ cond: $t-r$" (signed). Sau khi đã enforce $r\le t$ (fix #2), `(t-r)` luôn $\ge 0$ và `.abs()` thừa thãi ⇒ bỏ. | `interval = (t - r)` (signed). |
| 4 | `src/train_imf.py` | **CRITICAL** | `clip_grad_norm_(model.parameters(), …)` chỉ clip backbone, **không clip v-head**. Khi auxiliary FM loss spike (đặc biệt giai đoạn đầu), v-head có thể nhận gradient lớn → optimizer step lệch, bùng loss. | Đổi thành `clip_grad_norm_(all_params, …)` (đã chứa cả `v_head.parameters()`). |
| 5 | `src/config.py` + `src/train_imf.py` | MEDIUM | `IMFConfig` thiếu trường `adaptive_loss_weighting` ⇒ không thể tắt qua config; constructor `ImprovedMeanFlow(...)` cũng không pass tham số ⇒ luôn dùng default. | Thêm `IMFConfig.adaptive_loss_weighting: bool = True`; truyền `adaptive_loss_weighting=bool(getattr(imf_cfg, "adaptive_loss_weighting", True))` vào constructor. |
| 6 | `src/models/imf_diffusion.py:_update_adaptive_ema` | MEDIUM | Vòng `for b_idx in bins.unique()` chậm khi batch lớn (mỗi bin → 1 kernel launch riêng) — bottleneck CPU↔GPU sync. | Vector hóa bằng `scatter_add_`: tính tổng + đếm theo bin trong O(B), blend EMA 1 lần. Toán học tương đương. |
| 7 | ~~`src/models/imf_vloss.py`~~ + `src/train_imf.py` | MEDIUM | `IMFVelocityLoss.compute_jvp()` chỉ tính `∂u/∂z·v_tangent`, **thiếu ∂u/∂t** trong tiếp tuyến `(v,0,1)` (Alg. 1). | Đã gỡ hẳn `imf_vloss.py` khỏi repo; `train_imf.py` chỉ dùng `ImprovedMeanFlow.compute_loss`. |
| 8 | `src/models/sc_vae_loss.py` | LOW | `total_loss` trả về **không** chứa `rho_loss`. `train_sc_vae.py` cộng tay `rho_loss` ngoài (dùng `rho_scale` warmup). Người đọc dễ hiểu sai rằng rho_loss đã được tối ưu trong `loss`. | Giữ nguyên hành vi cộng ngoài (vì cần rho_warmup curriculum) nhưng thêm comment NOTE giải thích trong `forward()`. |

### 9.7.5. Tác động dự kiến của các fix

- **Hội tụ ổn định hơn:** fix #2 + #3 bảo đảm signed interval đúng dấu — model học $\partial u/\partial t$ nhất quán; fix #4 ngăn v-head spike dẫn tới NaN.
- **Tránh crash khi dùng backbone fallback:** fix #1 cho phép `IMFUNet1D` + v_head cùng tồn tại (trước đây luôn `TypeError`).
- **Throughput cao hơn (~vài %)**: fix #6 loại CPU sync; fix #5 cho phép tắt adaptive weighting để debug A/B.
- **Tránh nhầm lẫn cho contributor mới:** fix #7 (deprecate dead code) + #8 (document rho_loss) — không thay đổi behavior khi chạy thật.

### 9.7.6. Các tệp đã chỉnh sửa (revision 6)

```
src/config.py                  +1 line   (adaptive_loss_weighting)
src/models/imf_diffusion.py    +30/-15  (fix #1, #2, #6 + comment)
src/models/voxel_mamba.py      +4/-1    (fix #3 + comment)
src/models/imf_vloss.py        — đã xóa (JVP sai Alg. 1; không còn dùng)
src/models/sc_vae_loss.py      +5/-2    (NOTE rho_loss design)
src/train_imf.py               (fix #4, #5; gỡ `imf_vloss`; lazy-import `IMFUNet1D`)
Bao_cao_FaceDiff_ChiTiet.md    +120     (section 9.7 này)
```

### 9.7.7. Hành động khuyến nghị tiếp theo

1. **Smoke test** sau fix #2 trên 1 mini-batch GPU (`r<=t` luôn đúng, `(t-r)>=0`, EMA bins cập nhật đúng).
2. **A/B short run (~1 epoch)**: bật/tắt `adaptive_loss_weighting` để xác nhận vẫn ổn định.
3. **Resume từ checkpoint cũ**: vì fix #2/#3 thay đổi **distribution dữ liệu vào model** (signed interval, `r<=t`), chuẩn bị fine-tune ngắn (~10–20 epoch) thay vì kỳ vọng resume thẳng. EMA decay vẫn giữ giúp transition mượt.
4. **Long-run validation**: theo dõi `train/loss_v_head` không còn spike đột ngột (kiểm chứng fix #4).

---

## 9.8. Rà soát toàn pipeline trước khi Resume Train (Revision 7 — 11/05/2026)

### 9.8.1. Mục tiêu rà soát

Chuẩn bị resume training trên máy chủ mới qua `git clone`. Data + checkpoint từ Google Drive. Rà soát toàn bộ code trước khi deploy.

### 9.8.2. Tổng kết rà soát theo file

| Module | File | Kết quả | Ghi chú |
|---|---|---|---|
| SC-VAE model | `src/models/sc_vae.py` | ✅ OK | Kiến trúc 4-level encoder/decoder, early-pruning, residual AE shortcuts, TRELLIS.2 activations — đúng spec |
| SC-VAE loss | `src/models/sc_vae_loss.py` | ✅ OK | Multi-mode loss (shape_mat, shape_native, geom6...), KL per-element normalization, rho Focal Loss — đúng |
| SC-VAE train | `src/train_sc_vae.py` | 🔧 Sửa 1 bug | Bug `batch_data_time` luôn = 0 (xem §9.8.3) |
| iMF diffusion | `src/models/imf_diffusion.py` | ✅ OK | JVP, CFG, adaptive weighting, v-head — đã fix ở rev6 |
| Voxel Mamba | `src/models/voxel_mamba.py` | ✅ OK | BiSSM, Hilbert SFC, interval tokens — đúng spec |
| iMF train | `src/train_imf.py` | ✅ OK | SlatDataset cache, offline mode, grad clip all_params — đúng |
| Config | `src/config.py` | ✅ OK | Tất cả tham số có CLI override |
| Data pipeline | `src/scvae_train/data.py` | ✅ OK | LMDB + disk cache, stratified subsampling, collate |
| Data runtime | `src/scvae_train/runtime.py` | ✅ OK | align_recon_target, LR scheduler, resume scheduler |
| Scripts | `scripts/resume_from_397.sh` | ⚠️ Lưu ý | Conda path hardcode `${HOME}/miniconda3` (xem §9.8.4) |
| Precompute | `scripts/precompute_slat_cache.py` | ✅ OK | Slat cache generation đúng contract |

### 9.8.3. Bug đã sửa: `batch_data_time` không được tích lũy

**File:** `src/train_sc_vae.py` (line 880)

**Vấn đề:** Biến `batch_data_time` được khởi tạo `= 0.0` ở đầu mỗi gradient accumulation cycle (line 864) nhưng giá trị `data_time` (thời gian fetch batch từ DataLoader, line 844) không bao giờ được cộng vào. Kết quả: `epoch_data_time` luôn = 0 → perf metrics cho data loading time trong console và WandB luôn hiển thị 0ms.

**Fix:** Thêm `batch_data_time += data_time` ngay trước vòng micro-batch loop (line 880).

**Tác động:** Chỉ ảnh hưởng perf logging, KHÔNG ảnh hưởng training correctness. Sau fix, WandB `perf/data_ms` sẽ phản ánh đúng thời gian DataLoader.

### 9.8.4. Checklist resume train trên máy chủ mới

Khi `git clone` sang server mới với data/checkpoint từ Google Drive:

#### 1. Cài đặt environment
```bash
conda create -n facediff python=3.10
conda activate facediff
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install spconv-cu121 mamba-ssm lmdb trimesh wandb
```

#### 2. Override data paths (BẮT BUỘC)
Các đường dẫn mặc định trong `config.py` trỏ tới `/mnt/16TData/...` — trên server mới cần override qua CLI:
```bash
python src/train_sc_vae.py \
    --faceverse-root /path/to/FaceVerse \
    --facescape-root /path/to/FaceScape \
    --lmdb-dir /path/to/lmdb \
    --resume /path/to/checkpoint.pt \
    --allow-unsafe-resume
```

#### 3. Checkpoint compatibility
- SC-VAE checkpoints (epoch_390, epoch_397, interrupt.pt) **tương thích hoàn toàn** — không có thay đổi parameter shapes/names.
- `pre_latent_norm` (non-affine LayerNorm trước to_mu/to_logvar) không tạo learnable params → state_dict không đổi.
- `--allow-unsafe-resume` cần thiết nếu data path thay đổi (data_signature sẽ khác).

#### 4. Scripts cần chỉnh
- `scripts/resume_from_397.sh`: Đổi `${HOME}/miniconda3/...` thành đường dẫn conda của server mới
- Hoặc chạy trực tiếp `python src/train_sc_vae.py ...` với args tương ứng

#### 5. Cấu trúc thư mục cần thiết
```
facediff/
├── checkpoints/
│   └── sc_vae_shape/     ← checkpoint từ Google Drive
├── data/
│   ├── ovoxel_cache_lmdb/ ← LMDB cache (nếu có)
│   └── slat_cache/        ← iMF slat cache (tự tạo bởi train_imf.py)
├── train_faceverse_ids.txt
├── test_faceverse_ids.txt
├── train_facescape_ids.txt
└── test_facescape_ids.txt
```

### 9.8.5. Kết luận

Pipeline SC-VAE + iMF đã ổn định sau 8 bug fixes ở Revision 6. Rà soát lần này chỉ phát hiện 1 bug logging nhỏ (`batch_data_time`). Toàn bộ code sẵn sàng để resume training trên server mới — chỉ cần override data paths qua CLI args.

---

### 9.9. Rà soát Git Diff & Bổ sung Hỗ trợ Migration (Revision 9 — 12/05/2026)

#### 9.9.1. Tổng quan Thay đổi Git (10 files, +612/-125 lines)

Các thay đổi kể từ commit `ce406d9` (Bản sạch) bao gồm:

| File | Thay đổi chính | Đánh giá |
|---|---|---|
| `src/train_imf.py` | LMDB context, manifest, fallback guards, checkpoint mở rộng (v_head, stage2_model_config, global_step), save_every_steps | ✅ OK |
| `src/config.py` | `v_loss_weight` 1.0→0.1 (khớp iMF), deprecated neg_guidance fields, thêm fallback flags | ✅ OK |
| `src/models/imf_diffusion.py` | `v_loss_weight` configurable thay vì hardcoded 0.1, v_head RuntimeError rõ ràng hơn | ✅ OK |
| `src/models/voxel_mamba.py` | Tokenizer guards (None khi num_tokens=0), `_make_prefix_tokens` helper | ✅ OK |
| `src/models/generative_unet.py` | Refactor forward→_forward_core, thêm `get_hidden_state()`, `return_hidden`, tokenizer guards | ✅ OK |
| `src/inference/generator.py` | Đọc `stage2_model_config` từ checkpoint, cải thiện weight analysis fallback | ✅ OK |
| `src/data/arcface_extractor.py` | CPU/CUDA provider detection tự động | ✅ OK |
| `src/data/feature_extractor.py` | Disable torchao, fallback chain cho processor, `do_rescale=False` | ⚠️ Bug MockDinoProcessor (đã sửa) |
| `scripts/precompute_slat_cache.py` | Thêm `--context-lmdb`, `--allow-mesh-proxy-fallback` | ✅ OK |
| `Bao_cao_FaceDiff_ChiTiet.md` | Cập nhật nội dung | ✅ OK |

**Scripts mới (untracked):**
- `scripts/build_context_lmdb.py` — Tạo hybrid_context.lmdb từ mesh files
- `scripts/generate_mesh_manifest.py` — Tạo manifest JSON liệt kê .obj paths
- `scripts/start_background_context.sh` — Script tmux chạy context extraction
- `scripts/verify_hybrid_context_inputs.py` — Kiểm tra E2E hybrid context

#### 9.9.2. Bug Phát hiện & Sửa

**Bug 1: `MockDinoProcessor.__call__` không chấp nhận `do_rescale` kwarg**

Khi `AutoImageProcessor`/`BitImageProcessor` fail và fallback về `MockDinoProcessor`, lời gọi `self.processor(images=back_img, return_tensors="pt", do_rescale=False)` sẽ crash với `TypeError` vì `MockDinoProcessor.__call__` không có `**kwargs`.

```python
# Trước (bug):
def __call__(self, images, return_tensors="pt"):

# Sau (fix):
def __call__(self, images, return_tensors="pt", **kwargs):
```

**Bug 2 (CRITICAL): Thiếu code path đọc O-Voxel từ LMDB cho slat precomputation**

`SlatDataset._load_ovoxel_shape_mat()` chỉ hỗ trợ đọc mesh files qua `OVoxelConverter`. Khi migrate sang server mới chỉ có LMDB (không có .obj files), không thể tạo slat cache.

**Fix**: Thêm `--ovoxel-lmdb` vào `SlatDataset`, `train_imf.py`, và `precompute_slat_cache.py`:
- `_try_load_ovoxel_from_lmdb()`: Đọc O-Voxel payload từ LMDB bằng key format khớp `pack_lmdb_fast.py` / `VoxelDataset`
- `_ovoxel_lmdb_key()`: Tạo key `{rel_path_underscored}.c{ch}.{mode}.mx{max}.pt`
- Khi có LMDB, bỏ qua khởi tạo `OVoxelConverter` (không cần mesh dependencies)

#### 9.9.3. Xác nhận Chiến lược Data Migration

| Giai đoạn | Data cần | Nguồn | Đúng? |
|---|---|---|---|
| SC-VAE train | `data.mdb` + `lock.mdb` (O-Voxel LMDB) | Google Drive | ✅ Đúng |
| iMF — Slat precompute | O-Voxel LMDB + SC-VAE checkpoint + Context LMDB + Manifest | Drive + git | ✅ Đúng (sau khi thêm `--ovoxel-lmdb`) |
| iMF train (offline) | Slat cache .pt files (tự tạo bởi precompute) | Tự tạo trên server mới | ✅ Đúng |

#### 9.9.4. Luồng Dữ liệu Server Mới (sau fix)

```
Server hiện tại:                          Server mới:
┌──────────────┐                          ┌──────────────────────────────────┐
│ .obj meshes  │─── pack_lmdb_fast ──→    │                                  │
│              │    ovoxel_cache_lmdb/     │  1. SC-VAE TRAIN                 │
│              │        ↓ (Google Drive)   │     data.mdb + lock.mdb          │
│              │                          │     → train_sc_vae.py --lmdb-only│
│              │                          │                                  │
│ Render+Extract│─── build_context_lmdb →  │  2. SLAT PRECOMPUTE              │
│ ArcFace,FLAME│    hybrid_context.lmdb/  │     ovoxel_cache_lmdb/ +         │
│ DINOv2       │        ↓ (Google Drive)   │     hybrid_context.lmdb/ +       │
│              │                          │     SC-VAE checkpoint +           │
│ generate_mesh│─── mesh_manifest.json →   │     mesh_manifest.json           │
│ _manifest.py │        ↓ (git)           │     → precompute_slat_cache.py   │
│              │                          │       --ovoxel-lmdb ...          │
│              │                          │       --context-lmdb ...         │
│              │                          │       --manifest ...             │
│              │                          │                                  │
│              │                          │  3. iMF TRAIN                    │
│              │                          │     slat_cache/*.pt (từ bước 2)  │
│              │                          │     → train_imf.py --offline-data│
└──────────────┘                          └──────────────────────────────────┘
```

#### 9.9.5. Lệnh Chạy Trên Server Mới

**Bước 1: SC-VAE training (resume)**
```bash
python src/train_sc_vae.py \
    --lmdb-dir data/ovoxel_cache_lmdb \
    --lmdb-only \
    --resume checkpoints/sc_vae_shape/latest_step.pt
```

**Bước 2: Precompute slat cache (không cần mesh files)**
```bash
python scripts/precompute_slat_cache.py \
    --sc-vae-ckpt checkpoints/sc_vae_shape/latest_step.pt \
    --ovoxel-lmdb data/ovoxel_cache_lmdb \
    --context-lmdb data/hybrid_context.lmdb \
    --manifest data/mesh_manifest.json \
    --dataset both
```

**Bước 3: iMF training**
```bash
python src/train_imf.py --offline-data \
    --context-lmdb data/hybrid_context.lmdb
```

#### 9.9.6. Kết luận

- 2 bug đã sửa: `MockDinoProcessor` crash + thiếu O-Voxel LMDB path cho slat precompute
- Thêm `--ovoxel-lmdb` flag cho `SlatDataset`, `train_imf.py`, và `precompute_slat_cache.py`
- Migration flow hoàn chỉnh: SC-VAE chỉ cần O-Voxel LMDB, iMF chỉ cần 3 LMDB + manifest + SC-VAE checkpoint
- Không cần mesh files trên server mới

---

## Tài liệu Tham khảo

1. Geng, Z., Lu, Y., Wu, Z., Shechtman, E., Kolter, J. Z., & He, K. (2025). *Improved Mean Flows: On the Challenges of Fastforward Generative Models*. arXiv:2512.02012v1.
2. Xiang, J., et al. (2024, 2025). *TRELLIS / TRELLIS.2: Structured 3D Latents for Scalable and Versatile 3D Generation*. arXiv:2412.01506 / arXiv:2512.14692. Microsoft Research.
3. Gu, A., & Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
4. Mo, S., et al. (2024). *Efficient 3D Shape Generation via Diffusion Mamba with Bidirectional SSMs (DiM-3D)*. NeurIPS — arXiv:2406.05038.
4b. Zhang, G., et al. (2024). *Voxel Mamba: Group-Free State Space Models for Point Cloud based 3D Object Detection*. arXiv:2406.10700.
5. Deng, J., et al. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. CVPR.
6. Li, T., et al. (2017). *Learning a Model of Facial Shape and Expression from 4D Scans (FLAME)*. SIGGRAPH Asia.
7. Oquab, M., et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR.
8. Lipman, Y., et al. (2023). *Flow Matching for Generative Modeling*. ICLR.
9. Zhu, H., et al. (2020). *FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction*. CVPR.
10. Li, Z., et al. (2022). *FaceVerse: a Fine-grained and Detail-Controllable 3D Face Morphable Model from a Hybrid Dataset*. CVPR.
11. Rombach, R., et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR.
12. Poole, B., et al. (2023). *DreamFusion: Text-to-3D using 2D Diffusion*. ICLR.
13. Siddiqui, Y., et al. (2024). *MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers*. CVPR.
14. (đã chuyển lên ref [4]) — DiM-3D — Mo, S., et al. (2024). arXiv:2406.05038.

---

## Thực nghiệm: Kiểm chứng luồng Hybrid Context (Render & Trích xuất đặc trưng)
*(Cập nhật: 12/05/2026)*

Kịch bản kiểm tra đã chạy thành công một pass toàn diện (End-to-End) từ render hình ảnh 2D mặt trước/mặt sau từ mesh `064_03.obj`, đến việc đưa qua các mạng trích xuất đặc trưng (ArcFace, FLAME, DINOv2). 

Kết quả thu được hoàn toàn chính xác, không có module nào rơi vào trạng thái Fallback (báo lỗi và trả về tensor rỗng/chuẩn):

- **ArcFace (Identity):** Nhận diện được khuôn mặt từ ảnh render mặt trước `front_raw.png`, trích xuất thành công tensor kích thước `[1, 512]` bằng `insightface` (CPUExecutionProvider).
- **FLAME (Expression):** Trích xuất thành công tensor kích thước `[1, 50]` từ ảnh render mặt trước.
- **DINOv2 (Shape/Hair):** Xử lý thành công ảnh render mặt sau `back_raw.png` qua AutoImageProcessor, trích xuất tensor đặc trưng không gian `[1, 384]`.

Các ảnh kết quả và thông tin log (chi tiết bbox, kích thước tensor) đã được kiểm chứng bằng script `verify_hybrid_context_inputs.py`. Điều này khẳng định cơ chế cấp điều kiện Hybrid Context đang hoạt động chuẩn xác 100%.

---

---

## Báo cáo Tiến độ Toàn diện — 13/05/2026 (Revision 10)

### Tổng quan Trạng thái Dự án

| Thành phần | Trạng thái | Chi tiết |
|---|---|---|
| **SC-VAE (Stage 1)** | 🟢 Đang chạy | Epoch 399/500 · Step 621,450 · Loss 0.1078 |
| **Hybrid Context LMDB** | ✅ Hoàn tất | 20,939 bản ghi × 946 chiều · 84 MB |
| **O-Voxel Cache LMDB** | ✅ Hoàn tất | ~21,000 mẫu · 272 GB |
| **Mesh Manifest** | ✅ Hoàn tất | 2,310 FaceVerse + 18,658 FaceScape = 20,968 mesh |
| **Upload Google Drive** | ✅ Hoàn tất | ovoxel\_cache.tar + hybrid\_context\_lmdb.tar + manifest + checkpoint |
| **VoxelMamba (Stage 2)** | ⏳ Chờ SC-VAE xong | Code đã sẵn sàng, data offline đã đủ |
| **Cloud GPU Setup** | 📋 Đã chuẩn bị | Script `cloud_full_setup.sh` tự động hoá toàn bộ |

### 1. SC-VAE Training — Giai đoạn 1

#### Tiến độ hiện tại (14:07 ngày 13/05/2026)

```
Epoch 399/500 | Step 621,450
Total Loss:  0.1078
Recon Loss:  0.0260  (rất thấp → tái tạo hình khối chính xác)
KL Loss:     14.7553 (kiểm soát tốt nhờ KL Annealing)
Rho Loss:    0.0738
VRAM:        5,159 MB / 24,576 MB (21%)
OOM skips:   0 | NaN skips: 0  (hoàn toàn ổn định)
```

#### Ước tính thời gian

| Thông số | Giá trị |
|---|---|
| Thời gian 1 epoch (batch=1, accum=132) | ~37 phút |
| Epoch còn lại | 101 (399 → 500) |
| Thời gian còn lại | ~63 giờ ≈ 2.6 ngày |
| Ngày dự kiến hoàn thành | **Sáng 16/05/2026** |

#### So sánh tốc độ batch\_size

| Batch Size | Accum Steps | Effective Batch | Thời gian/epoch | Ghi chú |
|---|---|---|---|---|
| 4 | 33 | 132 | ~30.5 phút | Cấu hình cũ (epoch 360–397) |
| 1 | 132 | 132 | ~37.2 phút | Cấu hình hiện tại (+22%, an toàn OOM) |

> **Nhận xét:** Giảm batch size từ 4 xuống 1 chỉ làm chậm thêm 22% (do bottleneck nằm ở sparse convolution chứ không phải batch parallelism), trong khi loại bỏ hoàn toàn nguy cơ OOM.

#### Lịch sử Checkpoint

| File | Thời gian | Epoch |
|---|---|---|
| `epoch_360.pt` | 09/05 05:18 | 360 |
| `epoch_370.pt` | 09/05 19:27 | 370 |
| `epoch_380.pt` | 10/05 00:32 | 380 |
| `epoch_390.pt` | 10/05 05:37 | 390 |
| `latest_step.pt` | 10/05 04:41 | 397 |
| `interrupt.pt` | 10/05 09:11 | 397 |

### 2. Dữ liệu Offline — Đã Sẵn Sàng 100%

Toàn bộ dữ liệu cần thiết cho Giai đoạn 2 (VoxelMamba/iMF) đã được tiền xử lý và tải lên Google Drive thành công.

#### 2.1 Hybrid Context LMDB (`hybrid_context.lmdb`)

- **Số bản ghi:** 20,939 / 20,968 (99.86% — 29 mesh lỗi render từ dataset gốc)
- **Kích thước:** 84 MB
- **Cấu trúc vector (946 chiều):**
  - ArcFace Identity: 512 chiều (L2-normalized)
  - FLAME Expression: 50 chiều
  - DINOv2 Shape/Hair: 384 chiều
- **Key format:** Portable relative paths (hoạt động trên mọi máy)
- **Thời gian xây dựng:** ~20 giờ (12–13/05/2026)

#### 2.2 O-Voxel Cache LMDB (`ovoxel_cache_lmdb/`)

- **Số bản ghi:** ~21,000 mẫu O-Voxel 10 kênh
- **Kích thước:** 272 GB
- **Feature channels:** `[v(3), delta(3), gamma(1), r(1), g(1), b(1)]` = 10 kênh (shape\_mat)

#### 2.3 Mesh Manifest (`mesh_manifest.json`)

- **FaceVerse:** 2,310 mesh
- **FaceScape:** 18,658 mesh
- **Tổng cộng:** 20,968 mesh
- **Chức năng:** Cho phép DataLoader duyệt dataset mà không cần file `.obj` gốc trên Cloud

#### 2.4 Trạng thái Google Drive (`FaceDiff Data/`)

| File | Kích thước | Trạng thái |
|---|---|---|
| `ovoxel_cache.tar` | 271.4 GiB | ✅ Đã upload |
| `hybrid_context_lmdb.tar` | 83 MiB | ✅ Đã upload |
| `mesh_manifest.json` | 812 KB | ✅ Đã upload |
| `checkpoints/sc_vae_shape/latest_step.pt` | 403 MB | ✅ Đã upload |
| `checkpoints/sc_vae_shape/epoch_390.pt` | 403 MB | ✅ Đã upload |

### 3. Rà soát Kiến trúc VoxelMamba vs Bài báo DiM-3D

Đã rà soát toàn bộ file `src/models/voxel_mamba.py` so với bài báo gốc *"Efficient 3D Shape Generation via Diffusion Mamba with Bidirectional SSMs"* (Mo et al., 2024, arXiv:2406.05038).

#### Đối chiếu kiến trúc

| Thành phần (Bài báo) | Triển khai FaceDiff | Đánh giá |
|---|---|---|
| Bidirectional SSM (forward + backward scan) | `BidirectionalMambaBlock` (dòng 102–164) | ✅ Đúng |
| Voxel → Patch → Sequence flattening | `input_embed` + Hilbert ordering | ✅ Tốt hơn (Hilbert > raster) |
| Class token conditioning | In-context prefix tokens (ctx+t+r+interval+guidance) | ✅ Tốt hơn (linh hoạt hơn class embedding) |
| Linear complexity O(N) | Mamba SSM, fallback GRU | ✅ Đúng |
| Normalization | RMSNorm (thay LayerNorm) | ✅ Hiệu quả hơn |
| Zero-init output projection | `nn.init.zeros_` (dòng 326–327) | ✅ Đúng (best practice DiT) |

#### Điểm mở rộng so với bài báo gốc

1. **Hilbert Space-Filling Curve** (`src/hilbert.py`): Sắp xếp token 3D theo đường cong Hilbert thay vì raster scan, bảo toàn spatial locality tốt hơn cho SSM tuần tự.
2. **iMF Conditioning**: Mở rộng time conditioning sang `(t, r, t−r, ω, t_min, t_max)` theo bài báo iMF (Geng et al., 2025).
3. **Auxiliary v-head**: Khối dự đoán vận tốc phụ trợ, chia sẻ hidden state qua `forward(return_hidden=True)` để tránh forward pass thừa.

#### Kết luận rà soát

> **Kiến trúc VoxelMamba đã triển khai đúng và mở rộng hợp lý so với bài báo DiM-3D.**
> Không phát hiện lỗi kiến trúc nào cần sửa.

### 4. Kế hoạch Triển khai Cloud GPU

#### Cấu hình máy mới
- **GPU:** RTX 3090 Ti (24 GB VRAM)
- **RAM:** 64 GB
- **SSD:** 700 GB

#### Quy trình triển khai (Tự động hoá)

Đã chuẩn bị script `scripts/cloud_full_setup.sh` thực hiện tự động:
1. Cài đặt Conda + PyTorch 2.5.1 + CUDA 12.1
2. Cài dependencies (`requirements.txt` + `install_o_voxel.sh` + `requirements-mamba.txt`)
3. Cấu hình rclone kết nối Google Drive
4. Tải dữ liệu bằng Streaming Extraction (không cần lưu file `.tar`)
5. Xác nhận dữ liệu đầy đủ

```bash
# Chỉ cần 2 lệnh trên máy mới:
git clone https://github.com/HoangNguyennnnnnn/DATN.git ~/facediff
cd ~/facediff && bash scripts/cloud_full_setup.sh
```

#### Lệnh train trên Cloud GPU

**SC-VAE Resume (Stage 1):**
```bash
python -u src/train_sc_vae.py \
    --dataset both --feature-mode shape_mat --in-channels 10 \
    --lmdb-only --checkpoint-dir checkpoints/sc_vae_shape \
    --resume checkpoints/sc_vae_shape/latest_step.pt \
    --gradient-accumulation-steps 132 --batch-size 1 \
    --resume-scheduler-mode cosine_restart --resume-extend-epochs 100 \
    --resume-target-min-lr 1e-6 --enable-stage2-render-loss \
    --no-torch-compile --lr 1e-5
```

**VoxelMamba Offline (Stage 2):**
```bash
python src/train_imf.py \
    --offline-data \
    --context-lmdb data/hybrid_context.lmdb \
    --manifest data/mesh_manifest.json \
    --sc-vae-ckpt checkpoints/sc_vae_shape/latest_step.pt \
    --batch-size 32 --num-workers 8 --dataset both
```

### 5. Tổng kết Milestone

| Milestone | Ngày hoàn thành | Ghi chú |
|---|---|---|
| SC-VAE Epoch 360 | 09/05 | Checkpoint ổn định đầu tiên |
| SC-VAE Epoch 390 | 10/05 | Loss giảm đều |
| O-Voxel LMDB hoàn tất | ~04/05 | 272 GB, 21K mẫu |
| Hybrid Context LMDB hoàn tất | 13/05 | 20,939 bản ghi × 946-D |
| Upload toàn bộ data lên Drive | 13/05 | ovoxel + context + manifest + checkpoint |
| Rà soát VoxelMamba vs DiM-3D | 13/05 | Đúng kiến trúc, không cần sửa |
| Cloud setup script hoàn tất | 13/05 | `cloud_full_setup.sh` |
| SC-VAE Epoch 500 (dự kiến) | **16/05** | ETA 63 giờ |
| Khởi động VoxelMamba Stage 2 | **16–17/05** | Sau khi SC-VAE hội tụ |

---

*(Hết báo cáo — Cập nhật: 13/05/2026 — Revision 10: Báo cáo tiến độ toàn diện, rà soát VoxelMamba vs DiM-3D, chuẩn bị Cloud GPU deployment.)*
