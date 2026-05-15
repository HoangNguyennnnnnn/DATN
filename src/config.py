"""
Cấu hình Huấn luyện FaceDiff
===============================
Các giá trị mặc định cho pipeline 2 giai đoạn hiện tại trên RTX 4090 (24GB VRAM).

Đường dẫn chuẩn hiện tại:
- Giai đoạn 1: SC-VAE thống nhất trên `shape_mat` (10 kênh)
- Giai đoạn 2: iMF trên Slat latents, backbone mặc định `VoxelMamba`
- Các chế độ thử nghiệm/nhánh rẽ cũ vẫn được giữ lại cho các bài kiểm thử tùy chọn
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """Cấu hình Dataset."""
    faceverse_root: str = "/mnt/16TData/Datasets/FaceVerse_3D/FaceVerse"
    facescape_root: str = "/mnt/16TData/Datasets/FaceScape"
    
    # Chế độ Dataset (Sprint 17)
    active_dataset: str = "both"  # "facescape", "faceverse", hay "both"
    
    # Kết xuất (Rendering)
    image_size: int = 512
    max_voxels: int = 350000
    
    # O-Voxel
    voxel_resolution: int = 256        # 256³ cho training (512³ cho production)
    voxel_channels: int = 10           # v(3) + delta(3) + gamma(1) + rgb(3) = định dạng shape_mat
    
    # DataLoader (được sử dụng bởi tất cả các kịch bản huấn luyện)
    # LMDB ~272GB dữ liệu thực, file map_size pre-allocated ~429GB trên HDD /dev/sdb
    # (ổ cơ học, rotational=1). Các ràng buộc:
    # - ChunkedRandomSampler đọc LMDB theo cụm tuần tự (chunk_size=500),
    #   chuyển random seek 4KB (4.9 MB/s) thành sequential read (150+ MB/s)
    # - lmdb_readahead=True hợp lý vì ChunkedRandomSampler đảm bảo truy cập tuần tự
    # - persistent_workers=True giữ LMDB txn sống xuyên suốt các epochs
    # - prefetch_factor=4 đọc trước 4 batch, GPU không phải đợi
    num_workers: int = 8               # Tăng lên 8 để tận dụng đa nhân CPU giải nén O-Voxel
    pin_memory: bool = True            # Tăng tốc sao chép H2D
    prefetch_factor: int = 4           # Đọc trước 4 batch để GPU không bao giờ phải đợi
    persistent_workers: bool = True    # Giữ worker sống để tái sử dụng LMDB txn
    dataloader_timeout: int = 300      
    lmdb_dir: Optional[str] = "data/ovoxel_cache_lmdb" 
    lmdb_readahead: bool = True        # Đọc trước tuần tự, hiệu quả nhờ ChunkedRandomSampler
    

@dataclass
class SCVAEConfig:
    """
    Giai đoạn 1: Huấn luyện SC-VAE
    ========================
    Nén các đặc trưng hình học O-Voxel → Slat tokens (4096 × 32-dim).
    
    Chiến lược VRAM (Được tối ưu cho RTX 4090):
    - Kích thước lô (Batch size) 128 × 100K voxels = ~1.2 triệu điểm thưa/lô
    - Lấp đầy ~20-22GB VRAM để tối ưu hóa nhân Tensor Cores
    """
    # Kiến trúc (Thu nhỏ cho VRAM 24GB của RTX 4090, dành cho chân dung khuôn mặt)
    # Bài báo TRELLIS.2 sử dụng ~800M tham số cho các tài sản 3D tổng quát
    # Với chân dung khuôn mặt có tối đa 100K voxels: ~50-80M tham số là đủ
    # Hợp đồng đặc trưng đầu vào v4.1:
    # - shape_native: [v(3), delta(3), gamma(1)] = 7 kênh
    # - shape_mat (MẶC ĐỊNH): [v(3), delta(3), gamma(1), r(1), g(1), b(1)] = 10 kênh (chỉ hỗ trợ RGB cho khuôn mặt)
    # - geom6: [xyz(3), pháp tuyến(3)] = 6 kênh
    # - geom_mat12: [geom(6), pbr(6)] = 12 kênh
    in_channels: int = 10              # shape_mat: 7 kênh hình học + 3 kênh RGB
    input_feature_mode: str = "shape_mat"  # geom6|mat6|rgb3|shape_native|shape_mat|geom_mat12
    latent_dim: int = 32               # Mở rộng từ 16 để biểu diễn tốt hơn
    encoder_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])  # Cân đối cho RTX 4090
    num_res_blocks: int = 2            # Nhiều khối phần dư ở mỗi cấp giống như trong bài báo
    
    # Huấn luyện (Tối ưu hóa mức VRAM tối đa 24GB trên RTX 4090)
    # VRAM thực tế đo lường: lô=20 × 100K voxels chiếm ~21GB/24GB (89% mức sử dụng)
    # Hiệu suất GPU duy trì mức 90%+ với cấu hình này. Độ trễ dữ liệu 
    batch_size: int = 4                # Effective batch = 4 × 33 = 132 (gradient accumulation)
    num_epochs: int = 500              # Sẽ dừng sớm khi val loss bão hoà
    # Lưu ý: checkpoint epoch 388/390/397 chạy với learning_rate=1e-5 (xem
    # `resume_contract` của ckpt). Default 5e-5 phù hợp cho train-from-scratch.
    # Khi --resume từ checkpoint cũ, dùng --lr 1e-5 hoặc --resume-scheduler-mode
    # cosine_restart để tránh nhảy LR đột ngột.
    learning_rate: float = 5e-5        # HẠ LR (An toàn cho 350k points): Tránh nổ Loss khi KL Annealing tăng cao
    weight_decay: float = 0.0          # BỎ WEIGHT DECAY (Chuẩn TRELLIS.2)
    lr_warmup_steps: int = 500          # 500 steps (~3.4 epochs) — đủ ổn định cho cả train-from-scratch và fine-tune/resume
    lr_scheduler: str = "cosine_with_min_lr"  # Không giảm về không, tối thiểu 1e-6
    min_lr: float = 1e-6
    # Resume scheduler extension (xem train_sc_vae.py --resume-scheduler-mode).
    # Mặc định "continue" giữ tương thích với cosine cũ; "cosine_restart" và
    # "constant_min_lr" dành cho fine-tune sau khi cosine ban đầu đã chạy gần hết.
    resume_scheduler_mode: str = "continue"
    resume_extend_epochs: int = 100
    resume_target_min_lr: float = 1e-6
    
    # Trọng số mất mát (Đồng nhất với bài báo TRELLIS.2 Mục 3.2.2)
    kl_weight: float = 1e-6            # NỚI LỎNG KL (Chuẩn TRELLIS.2): Tăng chất lượng hình học tái tạo (Recon)
    kl_warmup_epochs: int = 20         # Ủ nhiệt KL (annealing) dài hơn (20 epoch) để tránh nổ Loss
    rho_loss_weight: float = 0.2       # Tăng lên 0.2 để khắc phục bệnh "mù" cờ giao cắt khi Resume từ 100k
    rho_warmup_epochs: int = 20        # Ủ nhiệt cho giám sát rho dài hơn để ổn định cấu trúc thưa
    rho_prune_threshold: float = 0.5   # Ngưỡng chiếm đóng của nút con cho quá trình cắt tỉa sớm
    use_bce_for_geom: bool = False     # Dùng MSE cho hình học (Pt. 6 trong bài báo)
    
    # Hàm mất mát kết xuất Giai đoạn 2 (Huấn luyện hai giai đoạn theo bài báo TRELLIS.2)
    # Giai đoạn 1 (0-49): Tái tạo trực tiếp O-Voxel - không gian đặc trưng thuần túy
    # Giai đoạn 2 (50+): Thêm giám sát cảm nhận dựa trên kết xuất để nâng cao chất lượng thị giác
    use_stage2_render_loss: bool = False      # OFF mặc định: LPIPS/render OOM trên 24GB GPU; bật bằng --enable-stage2-render-loss
    stage2_render_start_epoch: int = 50        # Chờ các đặc trưng hình dáng ổn định
    stage2_render_weight: float = 1.0          # Pt.(7) trong bài báo
    stage2_perceptual_weight: float = 0.2      # Theo bài báo: L1 + 0.2*SSIM + 0.2*LPIPS
    stage2_render_views: int = 2               # [PERF] 2 góc (front+side) đủ bao phủ khuôn mặt, tiết kiệm ~2.7%
    stage2_render_image_size: int = 64         # [PERF] LPIPS chuẩn ở 64px, tiết kiệm ~4% (scatter 4x nhỏ hơn)
    stage2_normal_weight: float = 1.0          # λ_normal: Trọng số pháp tuyến bề mặt từ depth (TRELLIS.2 standard)
    stage2_max_points_per_batch: int = 10000000 # CHẾ ĐỘ CHẤT LƯỢNG CAO: Giữ 100% điểm cho LPIPS loss
    
    # Các voxel cho mỗi lưới (Chiến lược VRAM tối đa)
    max_voxels_per_mesh: int = 350000   # NÂNG LÊN 350K: Bao phủ trọn vẹn cả những mẫu FaceVerse nặng nhất
    max_points_per_batch: int = 10000000 # CHẾ ĐỘ CHẤT LƯỢNG CAO: Không giới hạn điểm, 4090 nạp 100% mesh 350k
    

    # Xác thực (Validation)
    val_split: float = 0.05            # Giữ 5% tập huấn luyện để chọn điểm kiểm tra (checkpoint) tốt nhất
    val_every_epochs: int = 5          # [PERF] Giảm validation frequency, tiết kiệm ~10 tiếng tổng

    # Hợp đồng biểu diễn dữ liệu
    use_ovoxel_converter: bool = True   # Ưu tiên chuyển đổi O-Voxel thay vì lưới điểm (raw mesh points)
    ovoxel_resolution: int = 256        # Độ phân giải O-Voxel cho Giai đoạn 1
    require_ovoxel_converter: bool = True  # Báo lỗi nhanh (Fail fast) nếu bộ chuyển đổi O-Voxel không khả dụng
    
    # Điểm kiểm tra (Checkpoint)
    checkpoint_dir: str = "checkpoints/sc_vae_shape"  # thư mục cũ; hiện tại hợp nhất trên SC-VAE 10 kênh
    save_every_epochs: int = 10
    save_every_steps: int = 1000       # Tăng từ 200 lên để tránh hiện tượng tắc nghẽn chặn do HDD (HDD blocking stalls)
    resume_from: Optional[str] = None
    resume_model_only: bool = False    # False = load đầy đủ optimizer+scheduler để tránh KL bùng
    
    # Độ chính xác (Tối ưu cho RTX 4090)
    use_amp: bool = True               # Dùng bfloat16 để đạt tốc độ tự nhiên của chip Ada
    amp_dtype: str = "float16"         # Dùng float16 để spconv không bị KeyError; kẹp (clamp) bảo vệ sẽ lo phần overflow
    # Cắt Gradient (TRELLIS.2 Standard)
    use_adaptive_clip: bool = True     # Bật cắt gradient thích ứng (AdaptiveGradClipper)
    adaptive_clip_max_norm: float = 1.0 # Giá trị norm mục tiêu (nhân với percentile)
    adaptive_clip_percentile: float = 95.0 # Cắt các đỉnh đột biến vượt quá bách phân vị thứ 95
    grad_clip: float = 1.0             # Giá trị dự phòng nếu adaptive tắt
    
    # EMA (Exponential Moving Average) — TRELLIS.2 standard
    use_ema: bool = True               # Bật EMA cho model weights (VRAM: ~140MB cho 35M params)
    ema_decay: float = 0.9999          # Decay rate theo TRELLIS.2 (0.9999)

    # Tối ưu hóa bộ nhớ
    use_gradient_checkpointing: bool = False  # Vô hiệu hóa để tăng tốc (VRAM cho phép)
    clear_cache_every_n_batches: int = 0       # Vô hiệu hóa để giữ VRAM ổn định và ngăn việc sụt giảm 1.7GB

    # Kiểm tra an toàn khi thực thi
    require_spconv: bool = True        # Ràng buộc cứng - không cho phép tính năng thay thế (fallback)


@dataclass
class StructureConfig:
    """Giai đoạn 0/1: Sinh không gian tiềm ẩn cấu trúc thưa thớt (sparse latent structure)."""

    # Bố trí sự lấp đầy trong không gian token.
    slat_length: int = 4096             # Lưới không gian tiềm ẩn cấu trúc 16^3
    context_dim: int = 946              # Số chiều ngữ cảnh lai
    occupancy_threshold: float = 0.5    # Ngưỡng nhị phân hóa dự đoán lấp đầy

    # Kiến trúc mô hình.
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_context_tokens: int = 8
    dropout: float = 0.0

    # Dữ liệu/Giám sát.
    ovoxel_resolution: int = 256        # Độ phân giải O-Voxel đầu vào
    max_pos_weight: float = 25.0        # Giới hạn trọng số cho lớp dương để cân bằng nhãn BCE
    cache_dir: str = "data/structure_cache"

    # Huấn luyện.
    batch_size: int = 64
    num_epochs: int = 200
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    use_amp: bool = True
    grad_clip: float = 1.0

    # Điểm kiểm tra.
    checkpoint_dir: str = "checkpoints/structure_gen"
    save_every_epochs: int = 10
    resume_from: Optional[str] = None
    resume_model_only: bool = False


@dataclass
class IMFConfig:
    """
    Giai đoạn 2: Huấn luyện iMF U-Net
    ============================
    Mô hình sinh: nhiễu → Slat tokens, được điều kiện hóa bằng định danh ArcFace.
    
    Chiến lược VRAM:
    - Kích thước lô 48 × [4096, 32] Slat tokens + [946] ngữ cảnh
    - U-Net ~70M tham số + đạo hàm (gradients) → tổng ~20GB VRAM
    - An toàn khi chạy trên RTX 4090 24GB
    """
    # Kiến trúc (Tối ưu cho 20K mẫu khuôn mặt, RTX 4090)
    # VoxelMamba (default backbone): ~20.88M tham số đo thực tế (sum(p.numel())/1e6).
    # Hybrid U-DiT legacy backbone: ~45M tham số. Có thể mở rộng lên ~80M nếu bị underfit.
    input_dim: int = 32                # Số chiều Slat token - BẮT BUỘC KHỚP với SC-VAE latent_dim
    hidden_dims: List[int] = field(default_factory=lambda: [160, 320, 640])  # Tỉ lệ cân đối
    num_bottleneck_layers: int = 6      # Tăng từ 4 lên để tăng khả năng của khối attention
    context_dim: int = 946             # Ngữ cảnh Lai v4.1 (ArcFace 512 + FLAME 50 + DINOv2_Back 384)
    slat_length: int = 4096            # Số Slat tokens trên mỗi lưới
    
    # Kiến trúc Voxel Mamba (v5.0 - thay thế IMFUNet1D)
    use_voxel_mamba: bool = True         # Dùng VoxelMamba thay vì IMFUNet1D
    voxel_mamba_backend: str = "auto"    # auto|mamba|gru
    voxel_mamba_strict: bool = False     # Bằng True -> báo lỗi nếu yêu cầu mamba backend nhưng không khả dụng
    mamba_hidden_dim: int = 512          # Chiều ẩn cho các khối Mamba
    mamba_num_layers: int = 12           # Số lượng khối BidirectionalMambaBlocks
    mamba_d_state: int = 16              # Số chiều trạng thái SSM
    mamba_d_conv: int = 4                # Kích thước hạt nhân tích chập (kernel size)
    mamba_expand: int = 2                # Hệ số mở rộng
    mamba_num_context_tokens: int = 8    # In-context tokens cho điều kiện ngữ cảnh
    mamba_num_time_tokens: int = 4       # In-context tokens cho t
    mamba_num_r_tokens: int = 4          # In-context tokens cho r
    mamba_num_interval_tokens: int = 4   # In-context tokens cho (t-r) theo iMF appendix
    mamba_num_guidance_tokens: int = 4   # In-context tokens cho [omega, tmin, tmax]
    dual_branch: bool = False          # Chia iMF thành các nhánh hình học và vật liệu
    shape_sc_vae_checkpoint: Optional[str] = None      # Điểm kiểm tra VAE riêng biệt chuyên biệt cho hình học (tùy chọn)
    material_sc_vae_checkpoint: Optional[str] = None   # Điểm kiểm tra VAE riêng biệt chuyên biệt cho vật liệu (tùy chọn)
    shape_feature_mode: str = "shape_mat"          # shape_mat dùng chung cho 10-kênh
    material_feature_mode: str = "rgb3"               # none|rgb1|rgb3|mat6|geom_mat12
    shape_target_in_channels: int = 10                # shape_mat 10 kênh
    material_target_in_channels: int = 3               # train_imf sẽ căn chỉnh mục này phù hợp với material_feature_mode
    material_condition_source: str = "gt"             # gt|pred_detached
    material_condition_dropout: float = 0.1            # Tỉ lệ bỏ qua điều kiện hình dáng để tạo độ vững chắc
    material_loss_weight: float = 1.0                  # Trọng số cho hàm mất mát vận tốc của vật liệu
    
    # Tối ưu hóa DataLoader
    num_workers: int = 8               # Nạp dữ liệu song song
    prefetch_factor: int = 4           # Lô xử lý / worker
    pin_memory: bool = True              # Tăng tốc GPU transfer
    
    # Huấn luyện (Tối ưu hóa cho kiến trúc được mở rộng quy mô)
    batch_size: int = 48               # Giảm nhẹ để phù hợp [160,320,640] + 6 lớp (~70M tham số)
    num_epochs: int = 400              # Khuyên dùng Early stopping (giám sát mất mát trên tập xác thực)
    learning_rate: float = 2e-4        # Tăng LR lên 2e-4 bù đắp cho batch_size x2
    weight_decay: float = 1e-5
    lr_warmup_steps: int = 1000
    lr_scheduler: str = "cosine"
    
    # Đặc tả iMF (arXiv:2512.02012v1 — Geng et al., Improved Mean Flows)
    sigma_min: float = 1e-4            # Biên độ nhiễu tối thiểu
    ratio_r_neq_t: float = 0.5        # 50% mẫu dùng JVP (r≠t), 50% dùng điều kiện biên (r=t)
    t_sampler: str = "curriculum"      # Paper Tab.4: logit-normal(−0.4,1); curriculum = giai đoạn logit-normal + pha uniform
    t_loc: float = -0.4               # Trung bình logit-normal (Giai đoạn 1: thiên vị ở giữa)
    t_scale: float = 1.0              # Tỉ lệ logit-normal
    curriculum_switch_ratio: float = 0.6   # Chuyển sang Giai đoạn 2 (Đồng đều) ở mức 60% tiến trình
    curriculum_uniform_prob: float = 0.8   # Xác suất chọn giá trị đồng đều ở Giai đoạn 2 của tiến trình
    cfg_conditioning_enable: bool = True    # iMF Mục 4.2: học điều hướng linh hoạt dưới dạng điều kiện
    
    # iMF v5.0: v-loss cùng với khối phụ trợ v-head (Chỉ số cải thiện lợi nhuận ROI cao)
    use_v_loss: bool = True              # Dùng v-loss thay vì u-loss (huấn luyện ổn định hơn)
    use_auxiliary_v_head: bool = True    # Dùng khối v-head phụ trợ để dự đoán vận tốc cận biên
    v_head_dim: int = 512               # Chiều ẩn cho khối phụ trợ v-head
    v_loss_weight: float = 0.1           # Trọng số auxiliary v-head loss; khớp objective đang dùng trong iMF paper impl
    boundary_condition_ratio: float = 0.5  # Deprecated: dùng ratio_r_neq_t để suy ra boundary ratio thực tế
    cfg_omega_min: float = 1.0              # Cận dưới thang điều hướng (1.0 = không điều hướng)
    cfg_omega_max: float = 8.0              # Cận trên thang điều hướng
    cfg_omega_power_beta: float = 1.0       # Giá trị beta hàm lũy thừa cho p(omega) ~ omega^-beta
    cfg_context_dropout: float = 0.1        # Loại bỏ các ngữ cảnh điều kiện để học nhánh vô điều kiện ổn định
    cfg_interval_conditioning: bool = True  # Điều kiện hóa trên đoạn [tmin, tmax] giống trong phụ lục iMF
    adaptive_loss_weighting: bool = True    # iMF Appendix A: per-bin EMA reweighting cho main + v-head loss
    neg_guidance_enable: bool = False      # Deprecated: negative guidance training chưa được wire trong stage-2 runtime hiện tại
    neg_guidance_scale: float = 0.0        # Deprecated: inference-only path có hỗ trợ, training path chưa dùng
    neg_guidance_fallback: str = "random"  # Deprecated cùng neg_guidance_enable
    far_neg_min_candidates: int = 32       # Deprecated cùng neg_guidance_enable
    far_neg_pool_size: int = 1024          # Deprecated cùng neg_guidance_enable
    ema_decay: float = 0.9995          # Giảm xuống nhẹ để thích nghi tốt hơn với 20K mẫu
    use_ema: bool = True               # EMA cải thiện chất lượng lấy mẫu
    dropout: float = 0.05              # Bổ sung một chút dropout để tránh overfit đối với 20K mẫu
    
    # Điểm kiểm tra
    checkpoint_dir: str = "checkpoints/imf_unet"
    sc_vae_checkpoint: str = "checkpoints/sc_vae_shape/latest_step.pt"  # Checkpoint kết thúc Giai đoạn-1 (tự động dùng bản mới nhất)
    save_every_epochs: int = 20
    save_every_steps: int = 500        # Lưu checkpoint ở bước mới nhất để an toàn khi tiếp tục giữa epoch
    resume_from: Optional[str] = None
    resume_model_only: bool = False    # Bỏ qua trạng thái optimizer/scheduler khi khôi phục
    
    # Tối ưu hóa Dữ liệu luồng
    use_precomputed_data: bool = False # Nếu true, bỏ qua load DINO, Arcface, Flame, chạy offline.
    allow_random_context_fallback: bool = False  # Debug-only: cho phép cache/train với context ngẫu nhiên khi extractor lỗi
    allow_mesh_proxy_fallback: bool = False      # Debug-only: cho phép thay O-Voxel bằng mesh proxy khi converter lỗi
    
    # Độ chính xác
    use_amp: bool = True
    grad_clip: float = 1.0


@dataclass
class InferenceConfig:
    """Tùy chọn thiết lập dành riêng cho Suy luận trong việc trích xuất lưới và điều hướng."""
    imf_checkpoint: str = "checkpoints/imf_unet/best.pt"
    structure_checkpoint: Optional[str] = None
    sc_vae_checkpoint: Optional[str] = "checkpoints/sc_vae_shape/latest_step.pt"
    sc_vae_shape_checkpoint: Optional[str] = None
    sc_vae_material_checkpoint: Optional[str] = None
    enable_structure_stage: bool = False
    structure_threshold: float = 0.5
    feature_mode: str = "shape_mat"  # luồng giải mã 10-kênh thống nhất mặc định hiện tại
    cfg_scale: float = 1.0              # Tỉ lệ điều hướng linh hoạt iMF omega
    cfg_tmin: float = 0.0               # Cận dưới trong khoảng CFG
    cfg_tmax: float = 1.0               # Cận trên trong khoảng CFG
    neg_guidance_scale: float = 0.0     # >0 bật kết hợp điều hướng ngữ cảnh âm
    mesh_backend: str = "auto"         # auto|sparseflex|flexicubes|diffmc|marching_cubes
    enforce_dual_contouring: bool = False  # Nếu là True, các chế độ shape-native phải sử dụng DC và báo lỗi nếu dùng fallback
    mesh_smooth_sigma: float = 0.5      # Làm mịn bằng Gaussian trước khi trích xuất bề mặt đồng mức (iso-surface)
    ovoxel_resolution: int = 256        # Kích thước lưới O-Voxel dùng cho thuật toán dual contouring
    


@dataclass
class WandBConfig:
    """Ghi log Weights & Biases."""
    enabled: bool = False
    project: str = "facediff"
    entity: Optional[str] = None       # Tên team/user WandB
    run_name: Optional[str] = None     # Tự động khởi tạo nếu None
    log_every_steps: int = 100         # Tăng từ 10 lên nhằm giảm tần suất đồng bộ CPU-GPU
    log_images_every_epochs: int = 10  # Ghi hình các lưới mẫu
    tags: List[str] = field(default_factory=lambda: ["facediff", "3d-face"])


@dataclass 
class TrainConfig:
    """Master config kết hợp tất cả."""
    data: DataConfig = field(default_factory=DataConfig)
    sc_vae: SCVAEConfig = field(default_factory=SCVAEConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    imf: IMFConfig = field(default_factory=IMFConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    
    # Global
    seed: int = 42
    device: str = "cuda:0"
    
    def print_summary(self):
        print("=" * 60)
        print("  Cấu hình Huấn luyện FaceDiff")
        print("=" * 60)
        print(f"  Thiết bị: {self.device}")
        print(f"  Seed: {self.seed}")
        print(f"\n  [SC-VAE] batch={self.sc_vae.batch_size}, epochs={self.sc_vae.num_epochs}, "
              f"lr={self.sc_vae.learning_rate}, latent={self.sc_vae.latent_dim}")
        print(f"  [Struct] batch={self.structure.batch_size}, epochs={self.structure.num_epochs}, "
              f"lr={self.structure.learning_rate}, hidden={self.structure.hidden_dim}")
        print(f"  [iMF]    batch={self.imf.batch_size}, epochs={self.imf.num_epochs}, "
              f"lr={self.imf.learning_rate}, hidden={self.imf.hidden_dims}")
        print(f"  [WandB]  enabled={self.wandb.enabled}, project={self.wandb.project}")
        print("=" * 60)


if __name__ == "__main__":
    cfg = TrainConfig()
    cfg.print_summary()
