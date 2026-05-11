import hashlib
import os
import random

import torch
from torch.utils.data import Dataset

from src.data.ovoxel_converter import OVoxelConverter
from src.models.sc_vae import SPCONV_AVAILABLE
from src.utils import extract_identity_from_obj_path

# Bộ đệm toàn cục (Global cache) cho các môi trường LMDB để ngăn ngừa lỗi "already open" trong các thiết lập multi-dataset/multi-worker.
_LMDB_ENV_CACHE = {}

if SPCONV_AVAILABLE:
    import spconv.pytorch as spconv
else:
    spconv = None


def _stable_seed_from_text(text: str) -> int:
    """Tạo một seed số nguyên xác định (deterministic integer seed) từ văn bản."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _stable_subsample_indices(n: int, keep: int, key: str) -> torch.Tensor:
    """Tạo các chỉ số tập con xác định (deterministic subset indices) cho việc lấy mẫu voxel có giới hạn thân thiện với bộ nhớ đệm (cache-friendly)."""
    keep = int(max(1, min(keep, n)))
    if keep >= n:
        return torch.arange(n, dtype=torch.long)

    seed = _stable_seed_from_text(key)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randperm(n, generator=gen)[:keep]


def _spatial_stratified_indices(coords: torch.Tensor, keep: int, key: str) -> torch.Tensor:
    """Tạo tập con phân tầng không gian xác định (deterministic spatially-stratified subset) để bao phủ bề mặt tốt hơn."""
    n = int(coords.shape[0])
    keep = int(max(1, min(keep, n)))
    if keep >= n:
        return torch.arange(n, dtype=torch.long)

    c = coords.to(dtype=torch.int64, device="cpu")
    ratio = float(max(n, 1)) / float(max(keep, 1))
    stride = max(1, int(round(ratio ** (1.0 / 3.0))))

    base_mask = (
        (torch.remainder(c[:, 0], stride) == 0)
        & (torch.remainder(c[:, 1], stride) == 0)
        & (torch.remainder(c[:, 2], stride) == 0)
    )
    base_idx = base_mask.nonzero(as_tuple=False).reshape(-1)

    if base_idx.numel() == 0:
        return _stable_subsample_indices(n, keep, f"{key}|fallback")

    if int(base_idx.numel()) > keep:
        rel = _stable_subsample_indices(int(base_idx.numel()), keep, f"{key}|trim")
        out = base_idx[rel]
        return out.sort().values

    out = base_idx
    need = keep - int(out.numel())
    if need > 0:
        rem_idx = (~base_mask).nonzero(as_tuple=False).reshape(-1)
        if rem_idx.numel() > 0:
            rel = _stable_subsample_indices(int(rem_idx.numel()), min(need, int(rem_idx.numel())), f"{key}|fill")
            out = torch.cat([out, rem_idx[rel]], dim=0)

    if int(out.numel()) < keep:
        selected = torch.zeros(n, dtype=torch.bool)
        selected[out] = True
        extra_pool = (~selected).nonzero(as_tuple=False).reshape(-1)
        if extra_pool.numel() > 0:
            rel = _stable_subsample_indices(int(extra_pool.numel()), min(keep - int(out.numel()), int(extra_pool.numel())), f"{key}|extra")
            out = torch.cat([out, extra_pool[rel]], dim=0)

    if int(out.numel()) > keep:
        rel = _stable_subsample_indices(int(out.numel()), keep, f"{key}|final")
        out = out[rel]

    return out.sort().values


def _shape_topology_subsample_indices(coords: torch.Tensor, features: torch.Tensor, keep: int, key: str) -> torch.Tensor:
    """Ưu tiên các voxel có độ kết nối cao, sau đó lấp đầy bằng độ bao phủ phân tầng không gian."""
    n = int(coords.shape[0])
    keep = int(max(1, min(keep, n)))
    if keep >= n:
        return torch.arange(n, dtype=torch.long)

    if not (isinstance(features, torch.Tensor) and features.ndim == 2 and features.shape[1] >= 6):
        return _spatial_stratified_indices(coords, keep, key)

    delta_score = features[:, 3:6].to(dtype=torch.float32).abs().sum(dim=1)
    important = (delta_score >= 2.0).nonzero(as_tuple=False).reshape(-1)

    if int(important.numel()) >= keep:
        rel = _stable_subsample_indices(int(important.numel()), keep, f"{key}|important")
        return important[rel].sort().values

    selected = important
    selected_mask = torch.zeros(n, dtype=torch.bool)
    if selected.numel() > 0:
        selected_mask[selected] = True

    need = keep - int(selected.numel())
    if need > 0:
        rem_pool = (~selected_mask).nonzero(as_tuple=False).reshape(-1)
        if rem_pool.numel() > 0:
            rem_coords = coords[rem_pool]
            rem_pick_rel = _spatial_stratified_indices(rem_coords, min(need, int(rem_pool.numel())), f"{key}|spatial")
            rem_pick = rem_pool[rem_pick_rel]
            selected = torch.cat([selected, rem_pick], dim=0)

    if int(selected.numel()) < keep:
        selected_mask = torch.zeros(n, dtype=torch.bool)
        if selected.numel() > 0:
            selected_mask[selected] = True
        extra_pool = (~selected_mask).nonzero(as_tuple=False).reshape(-1)
        if extra_pool.numel() > 0:
            rel = _stable_subsample_indices(int(extra_pool.numel()), min(keep - int(selected.numel()), int(extra_pool.numel())), f"{key}|extra")
            selected = torch.cat([selected, extra_pool[rel]], dim=0)

    if int(selected.numel()) > keep:
        rel = _stable_subsample_indices(int(selected.numel()), keep, f"{key}|cap")
        selected = selected[rel]

    return selected.sort().values


def _atomic_torch_save(payload, path: str) -> None:
    """Nỗ lực tốt nhất để lưu torch.save nguyên tử nhằm tránh các tệp bộ đệm bị lưu dở dang (partial cache files)."""
    tmp_path = f"{path}.tmp.{os.getpid()}.{random.randint(0, 10**9)}"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _is_valid_cached_payload(payload) -> bool:
    """Xác thực payload bộ đệm để tránh tái sử dụng các tệp hỏng hoặc trống rỗng."""
    if isinstance(payload, torch.Tensor):
        return payload.ndim == 2 and payload.shape[0] > 0 and payload.shape[1] > 0
    if isinstance(payload, dict) and "features" in payload:
        feats = payload.get("features", None)
        return isinstance(feats, torch.Tensor) and feats.ndim == 2 and feats.shape[0] > 0 and feats.shape[1] > 0
    return False


def _infer_cache_resolution(payload_resolution, coords, default_resolution: int) -> int:
    """Phân giải độ phân giải lưới voxel từ siêu dữ liệu (metadata) của payload với các cơ chế dự phòng an toàn (safe fallbacks)."""
    resolution = None

    if isinstance(payload_resolution, torch.Tensor) and payload_resolution.numel() > 0:
        resolution = int(payload_resolution.reshape(-1)[0].item())
    elif isinstance(payload_resolution, (int, float)):
        resolution = int(payload_resolution)
    elif hasattr(payload_resolution, "item"):
        try:
            resolution = int(payload_resolution.item())
        except Exception:
            resolution = None

    max_coord = None
    if isinstance(coords, torch.Tensor) and coords.numel() > 0:
        max_coord = int(coords.max().item())

    if resolution is None or resolution < 2:
        if max_coord is not None:
            resolution = max_coord + 1
        else:
            resolution = int(default_resolution)

    if max_coord is not None and max_coord >= resolution:
        resolution = max_coord + 1

    return int(max(2, resolution))


class VoxelDataset(Dataset):
    """Dataset trả về đặc trưng O-Voxel (O-Voxel features) cho từng lưới (mesh)."""

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        max_voxels: int = 10000,
        cache_dir: str = "data/voxel_cache",
        device: str = "cpu",
        use_ovoxel_converter: bool = False,
        ovoxel_resolution: int = 256,
        require_ovoxel_converter: bool = False,
        target_in_channels: int = 6,
        feature_mode: str = "geom6",
        include_ids=None,
        exclude_ids=None,
        lmdb_dir: str = None,
        lmdb_readahead: bool = True,
        lmdb_only: bool = False,
    ):
        self.data_root = data_root
        self.dataset_name = str(dataset_name)
        self.max_voxels = max_voxels
        self.cache_dir = cache_dir
        self.device = device
        self.use_ovoxel_converter = use_ovoxel_converter
        self.ovoxel_resolution = ovoxel_resolution
        self.require_ovoxel_converter = require_ovoxel_converter
        self.target_in_channels = int(target_in_channels)
        self.feature_mode = str(feature_mode)
        self.include_ids = include_ids
        self.exclude_ids = exclude_ids
        if self.target_in_channels <= 0:
            raise ValueError("target_in_channels must be > 0")
        if self.feature_mode not in {"geom6", "geom_mat12", "mat6", "rgb3", "shape_native", "shape_mat"}:
            raise ValueError(
                f"Unsupported feature_mode={self.feature_mode}. "
                "Use 'geom6', 'geom_mat12', 'mat6', 'rgb3', 'shape_native', or 'shape_mat'."
            )
        if self.feature_mode == "shape_native" and self.target_in_channels != 7:
            print(
                f"[VoxelDataset] shape_native expects 7 channels, "
                f"overriding target_in_channels {self.target_in_channels} -> 7"
            )
            self.target_in_channels = 7
        self.samples = []
        self.ovoxel_converter = None
        self.lmdb_dir = lmdb_dir
        self.lmdb_readahead = bool(lmdb_readahead)
        self.lmdb_only = bool(lmdb_only) and bool(lmdb_dir)
        self.lmdb_env = None
        self.lmdb_txn = None
        self._lmdb_miss_count = 0
        self._heavy_cap_warn_count = 0
        self._invalid_feature_warn_count = 0

        if bool(lmdb_only) and not bool(lmdb_dir):
            print("[VoxelDataset] Warning: lmdb_only requested but lmdb_dir is empty. Falling back to disk cache mode.")

        os.makedirs(cache_dir, exist_ok=True)

        if self.use_ovoxel_converter:
            try:
                self.ovoxel_converter = OVoxelConverter(
                    resolution=ovoxel_resolution,
                    device="cpu",
                )
                print(
                    f"[VoxelDataset] O-Voxel converter enabled "
                    f"(resolution={ovoxel_resolution})."
                )
            except Exception as e:
                if self.require_ovoxel_converter:
                    raise RuntimeError(
                        "O-Voxel converter initialization failed while "
                        "require_ovoxel_converter=True."
                    ) from e
                print(
                    "[VoxelDataset] Warning: O-Voxel converter unavailable, "
                    f"fallback to mesh-point features. Error: {e}"
                )

        skipped_by_include = 0
        skipped_by_exclude = 0
        if os.path.isdir(data_root):
            for root_dir, _, files in os.walk(data_root):
                for f in sorted(files):
                    if f.endswith('.obj'):
                        obj_path = os.path.join(root_dir, f)
                        identity = extract_identity_from_obj_path(obj_path, self.data_root, self.dataset_name)
                        if self.include_ids is not None and identity not in self.include_ids:
                            skipped_by_include += 1
                            continue
                        if self.exclude_ids is not None and identity in self.exclude_ids:
                            skipped_by_exclude += 1
                            continue
                        self.samples.append(obj_path)

        print(f"[VoxelDataset] Found {len(self.samples)} meshes from {data_root}")
        if self.include_ids is not None or self.exclude_ids is not None:
            print(
                f"[VoxelDataset] Filtered: include_skip={skipped_by_include}, "
                f"exclude_skip={skipped_by_exclude}"
            )
        
        # QUAN TRỌNG: Sắp xếp các mẫu theo khóa LMDB để các chỉ số (indices) dataset tuần tự
        # ánh xạ tới các trang (pages) B-tree LMDB tuần tự. Điều này giúp ChunkedRandomSampler
        # đọc các trang vật lý nằm gần nhau trên đĩa HDD, chuyển đổi việc tìm kiếm ngẫu nhiên (random seeks)
        # 4KB (4.9 MB/s) thành đọc tuần tự (sequential reads) (150+ MB/s).
        if self.lmdb_dir:
            def _lmdb_sort_key(obj_p):
                rel = os.path.relpath(obj_p, self.data_root)
                return rel.replace(os.path.sep, '_').replace(
                    '.obj', f'.c{self.target_in_channels}.{self.feature_mode}.mx{self.max_voxels}.pt'
                )
            self.samples.sort(key=_lmdb_sort_key)
            print(f"[VoxelDataset] Sorted {len(self.samples)} samples by LMDB key for sequential I/O")
        
        if int(self.max_voxels) > 0:
            print(
                f"[VoxelDataset] Compact capped cache enabled: "
                f"max_voxels={int(self.max_voxels)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        obj_path = self.samples[idx]

        rel_path = os.path.relpath(obj_path, self.data_root)
        safe_name = rel_path.replace(os.path.sep, '_').replace('.obj', f'.c{self.target_in_channels}.{self.feature_mode}.pt')
        cache_path = os.path.join(self.cache_dir, safe_name)
        max_vox = int(max(0, self.max_voxels))
        capped_cache_path = None
        capped_name = None
        if max_vox > 0:
            capped_name = safe_name.replace('.pt', f'.mx{max_vox}.pt')
            capped_cache_path = os.path.join(self.cache_dir, capped_name)

        voxels = None
        loaded_from_capped_cache = False

        # Trong chế độ LMDB, ưu tiên LMDB trước tiên, nhưng tái sử dụng bất kỳ bộ đệm
        # thu gọn (compact cache) nào đã được vật chất hóa cục bộ để tránh việc phải giới hạn (re-capping) lại các payload thô lớn mỗi epoch.
        prefer_lmdb_first = not (
            capped_cache_path is not None and os.path.exists(capped_cache_path)
        )

        if (not prefer_lmdb_first) and capped_cache_path is not None and os.path.exists(capped_cache_path):
            try:
                voxels = torch.load(capped_cache_path, map_location="cpu", weights_only=True)
                if not _is_valid_cached_payload(voxels):
                    raise RuntimeError("Invalid cached payload (empty features)")
                loaded_from_capped_cache = True
            except Exception:
                try:
                    os.remove(capped_cache_path)
                except OSError:
                    pass
                voxels = None

        if voxels is None and self.lmdb_dir is not None:
            if self.lmdb_env is None:
                import lmdb
                
                # Sử dụng bộ đệm toàn cục (global cache) để ngăn ngừa lỗi "already open" trong cùng một tiến trình
                global _LMDB_ENV_CACHE
                if self.lmdb_dir not in _LMDB_ENV_CACHE:
                    env = lmdb.open(
                        self.lmdb_dir,
                        readonly=True,
                        lock=False,
                        readahead=self.lmdb_readahead,
                        meminit=False,
                        max_readers=512,
                    )
                    # Mở giao dịch (transaction) một lần cho mỗi tiến trình/worker
                    txn = env.begin(write=False)
                    _LMDB_ENV_CACHE[self.lmdb_dir] = (env, txn)
                
                self.lmdb_env, self.lmdb_txn = _LMDB_ENV_CACHE[self.lmdb_dir]

            lmdb_keys = []
            if capped_name is not None:
                lmdb_keys.append(capped_name)
            lmdb_keys.append(safe_name)

            import io
            for lmdb_key in lmdb_keys:
                lmdb_data = self.lmdb_txn.get(lmdb_key.encode('utf-8'))
                if lmdb_data is None:
                    continue
                try:
                    voxels = torch.load(io.BytesIO(lmdb_data), map_location="cpu", weights_only=False)
                    if not _is_valid_cached_payload(voxels):
                        raise RuntimeError("Invalid LMDB payload (empty features)")
                    if capped_name is not None and lmdb_key == capped_name:
                        loaded_from_capped_cache = True
                    break
                except Exception as e:
                    print(f"\n[CẢNH BÁO] LMDB Cache hỏng: {lmdb_key}. Trở về đọc Disk... Lỗi: {e}")

        if (
            voxels is None
            and (not self.lmdb_only)
            and capped_cache_path is not None
            and os.path.exists(capped_cache_path)
        ):
            try:
                voxels = torch.load(capped_cache_path, map_location="cpu", weights_only=True)
                if not _is_valid_cached_payload(voxels):
                    raise RuntimeError("Invalid cached payload (empty features)")
                loaded_from_capped_cache = True
            except Exception:
                try:
                    os.remove(capped_cache_path)
                except OSError:
                    pass
                voxels = None

        if voxels is None and (not self.lmdb_only) and os.path.exists(cache_path):
            try:
                voxels = torch.load(cache_path, map_location="cpu", weights_only=False)
                if not _is_valid_cached_payload(voxels):
                    raise RuntimeError("Invalid cached payload (empty features)")
            except Exception:
                print(f"\n[CẢNH BÁO] Cache hỏng (Corrupted): {safe_name}. Đang xoá và tái tạo lại...")
                os.remove(cache_path)

        if voxels is None:
            if self.lmdb_only and self.lmdb_dir is not None:
                self._lmdb_miss_count += 1
                if self._lmdb_miss_count <= 5 or (self._lmdb_miss_count % 200) == 0:
                    print(
                        f"[VoxelDataset] LMDB miss #{self._lmdb_miss_count}: {capped_name or safe_name}. "
                        "Falling back to mesh conversion (disk cache probes disabled)."
                    )
            voxels = self._convert_mesh(obj_path)
            if capped_cache_path is None and (not self.lmdb_only):
                _atomic_torch_save(voxels, cache_path)

        if isinstance(voxels, torch.Tensor):
            voxels = {
                "features": voxels,
                "coords": None,
                "aabb": None,
                "resolution": int(self.ovoxel_resolution),
            }

        features = voxels["features"].to(dtype=torch.float32)
        coords = voxels.get("coords", None)
        aabb = voxels.get("aabb", None)
        if coords is not None:
            coords = coords.to(dtype=torch.int32)
        if isinstance(aabb, torch.Tensor):
            aabb = aabb.to(dtype=torch.float32)
        else:
            aabb = None
        resolution = _infer_cache_resolution(voxels.get("resolution", None), coords, self.ovoxel_resolution)

        if not torch.isfinite(features).all():
            bad_count = int((~torch.isfinite(features)).sum().item())
            if self._invalid_feature_warn_count < 5 or (self._invalid_feature_warn_count % 50) == 0:
                print(
                    f"[VoxelDataset] Warning: non-finite features in {safe_name} "
                    f"({bad_count} values). Replacing with zeros."
                )
            self._invalid_feature_warn_count += 1
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        n = features.shape[0]
        if max_vox > 0 and n > max_vox:
            drop_ratio = 1.0 - (float(max_vox) / float(max(n, 1)))
            if (
                self.feature_mode in {"shape_native", "shape_mat"}
                and coords is not None
                and int(coords.shape[0]) == int(n)
            ):
                indices = _shape_topology_subsample_indices(coords, features, max_vox, safe_name)
                warn_limit = 1 if self.lmdb_only else 3
                if drop_ratio >= 0.40 and self._heavy_cap_warn_count < warn_limit:
                    print(
                        f"[VoxelDataset] Heavy cap on {safe_name}: {n} -> {max_vox} "
                        f"({drop_ratio * 100:.1f}% drop). Using topology-aware sampling for shape stability."
                    )
                    self._heavy_cap_warn_count += 1
            else:
                indices = _stable_subsample_indices(int(n), max_vox, safe_name)
            features = features[indices]
            if coords is not None and int(coords.shape[0]) >= int(indices.shape[0]):
                coords = coords[indices]

        if (
            capped_cache_path is not None
            and not loaded_from_capped_cache
            and not os.path.exists(capped_cache_path)
        ):
            compact_payload = {
                "features": features.to(dtype=torch.float16, copy=False),
                "coords": coords,
                "aabb": aabb,
                "resolution": int(resolution),
            }
            try:
                _atomic_torch_save(compact_payload, capped_cache_path)
            except Exception:
                pass

        # Xử lý việc cắt lớp (slicing) bộ đệm hợp nhất (Shape-only vs Material-only vs Full).
        # Bố cục 10 kênh từ ARCHITECTURE.md: [Shape 7ch, Material 3ch].
        if features.shape[1] >= 10:
            if self.feature_mode == "shape_native":
                features = features[:, :7]
            elif self.feature_mode == "mat6" or self.feature_mode == "mat_only":
                # Backward-compatible 6-channel material layout: [rgb(3), m(1), r(1), a(1)].
                rgb = features[:, 7:10]
                metallic = torch.zeros((rgb.shape[0], 1), dtype=rgb.dtype, device=rgb.device)
                roughness = torch.full((rgb.shape[0], 1), 0.5, dtype=rgb.dtype, device=rgb.device)
                alpha = torch.ones((rgb.shape[0], 1), dtype=rgb.dtype, device=rgb.device)
                features = torch.cat([rgb, metallic, roughness, alpha], dim=-1)
            elif self.feature_mode == "rgb3":
                # RGB-only material branch requested by current pipeline.
                features = features[:, 7:10]
            elif self.feature_mode == "shape_mat":
                # Keep full unified channels for cache/export path.
                pass

        return {
            "features": features,
            "coords": coords,
            "aabb": aabb,
            "resolution": int(resolution),
        }

    def _convert_mesh(self, obj_path: str) -> dict:
        if self.ovoxel_converter is not None:
            try:
                ovoxel = self.ovoxel_converter.process_mesh(obj_path)

                if isinstance(ovoxel, dict):
                    if 'shape_mat_features' in ovoxel and ovoxel['shape_mat_features'] is not None:
                        shape_mat = ovoxel['shape_mat_features']
                        geom = shape_mat[:, :7] # [v, delta, gamma]
                        mat = shape_mat[:, 7:]  # [rgb]
                        full = shape_mat
                    elif 'geom_features' in ovoxel:
                        geom = ovoxel['geom_features']
                        mat = ovoxel.get('mat_features', None)
                        full = ovoxel.get('full_features', None)
                    elif 'shape_features' in ovoxel:
                        geom = ovoxel['shape_features']
                        mat = ovoxel.get('mat_features', None)
                        full = ovoxel.get('full_features', None)
                    else:
                        raise KeyError("Missing shape_mat_features/geom_features in O-Voxel output")
                    
                    coords = ovoxel.get('coords', None)
                    aabb = ovoxel.get('aabb', None)
                else:
                    raise TypeError("Unexpected O-Voxel output type")

                geom_t = torch.as_tensor(geom, dtype=torch.float32, device='cpu')
                mat_t = None
                if mat is not None:
                    mat_t = torch.as_tensor(mat, dtype=torch.float32, device='cpu')

                # LƯU Ý: các đặc trưng shape_native[:,:3] là ĐỘ LỆCH (OFFSETS) của đỉnh kép (phạm vi ~[0,1])
                # bên trong mỗi ô voxel, KHÔNG PHẢI là tọa độ tuyệt đối. Việc chuẩn hóa chúng
                # sẽ phá hủy biểu diễn O-Voxel.
                normalize_xyz = self.feature_mode in {"geom6", "geom_mat12"}
                if self.feature_mode == "geom_mat12":
                    if full is not None:
                        base_features = full
                    else:
                        if mat is None:
                            if self.require_ovoxel_converter:
                                raise RuntimeError("mat_features unavailable for geom_mat12 mode")
                            mat_t = torch.zeros((geom_t.shape[0], 6), dtype=geom_t.dtype)
                        base_features = torch.cat([geom_t, mat_t], dim=-1)
                elif self.feature_mode == "mat6":
                    if mat_t is None:
                        if self.require_ovoxel_converter:
                            raise RuntimeError("mat_features unavailable for mat6 mode")
                        mat_t = torch.full((geom_t.shape[0], 6), 0.5, dtype=geom_t.dtype)
                        mat_t[:, 3] = 0.0
                        mat_t[:, 4] = 0.5
                        mat_t[:, 5] = 1.0
                    base_features = mat_t
                    normalize_xyz = False
                elif self.feature_mode == "rgb3":
                    if 'shape_mat_features' in ovoxel and ovoxel['shape_mat_features'] is not None:
                        # RGB-only material channels.
                        base_features = ovoxel['shape_mat_features'][:, 7:10]
                    elif mat_t is not None and mat_t.shape[1] >= 3:
                        base_features = mat_t[:, :3]
                    else:
                        if self.require_ovoxel_converter:
                            raise RuntimeError("material RGB unavailable for rgb3 mode")
                        base_features = torch.full((geom_t.shape[0], 3), 0.5, dtype=geom_t.dtype)
                    normalize_xyz = False
                elif self.feature_mode == "shape_native":
                    if 'shape_mat_features' in ovoxel and ovoxel['shape_mat_features'] is not None:
                        # Slice 7 channels: v(3), delta(3), gamma(1)
                        base_features = ovoxel['shape_mat_features'][:, :7]
                    elif 'shape_features' in ovoxel and ovoxel['shape_features'] is not None:
                        base_features = ovoxel['shape_features']
                    else:
                        if self.require_ovoxel_converter:
                            raise RuntimeError("shape features unavailable for shape_native mode")
                        base_features = geom
                elif self.feature_mode == "shape_mat":
                    if 'shape_mat_features' in ovoxel and ovoxel['shape_mat_features'] is not None:
                        # Keep unified 10-channel representation [shape(7), material(3)].
                        base_features = ovoxel['shape_mat_features']
                    else:
                        raise RuntimeError("shape_mat_features unavailable for shape_mat mode")
                elif self.feature_mode == "mat_only": # New mode for 10ch cache
                    if 'shape_mat_features' in ovoxel and ovoxel['shape_mat_features'] is not None:
                        # Slice 3 channels: rgb(3)
                        base_features = ovoxel['shape_mat_features'][:, 7:10]
                    else:
                        raise RuntimeError("shape_mat_features unavailable for mat_only mode")
                    normalize_xyz = False
                else:
                    base_features = geom

                features = torch.as_tensor(base_features, dtype=torch.float32, device='cpu')
                if features.ndim != 2:
                    raise ValueError(f"Expected [N, C] features, got shape {tuple(features.shape)}")

                if features.shape[1] < self.target_in_channels:
                    pad = torch.zeros(features.shape[0], self.target_in_channels - features.shape[1], dtype=features.dtype)
                    features = torch.cat([features, pad], dim=-1)
                elif features.shape[1] > self.target_in_channels:
                    features = features[:, :self.target_in_channels]

                if normalize_xyz and features.shape[1] >= 3:
                    center = features[:, :3].mean(dim=0)
                    features[:, :3] -= center
                    scale = features[:, :3].abs().max() + 1e-8
                    features[:, :3] /= scale

                coords_tensor = None
                if coords is not None:
                    coords_tensor = torch.as_tensor(coords, dtype=torch.int32, device='cpu')
                    if coords_tensor.ndim != 2 or coords_tensor.shape[1] != 3:
                        coords_tensor = None

                if coords_tensor is None:
                    if geom_t is not None and geom_t.shape[1] >= 3:
                        xyz = geom_t[:, :3]
                        if xyz.max().item() <= 1.0 and xyz.min().item() >= 0.0:
                            xyz = xyz * 2.0 - 1.0
                        else:
                            xyz = xyz.clone()
                            xyz_center = xyz.mean(dim=0)
                            xyz -= xyz_center
                            xyz_scale = xyz.abs().max() + 1e-8
                            xyz /= xyz_scale
                    elif features.shape[1] >= 3 and normalize_xyz:
                        xyz = features[:, :3]
                    else:
                        xyz = torch.rand((features.shape[0], 3), dtype=features.dtype) * 2.0 - 1.0
                    xyz = xyz.clamp(-1.0, 1.0)
                    coords_tensor = ((xyz + 1.0) * 0.5 * (self.ovoxel_resolution - 1)).to(torch.int32)
                    coords_tensor = torch.clamp(coords_tensor, 0, self.ovoxel_resolution - 1)

                return {
                    "features": features,
                    "coords": coords_tensor,
                    "aabb": aabb,
                    "resolution": int(self.ovoxel_resolution),
                }
            except Exception as e:
                # THAY ĐỔI ĐỘT PHÁ (BREAKING): Không cho phép âm thầm dự phòng (silent fallback) về đám mây điểm.
                # Nếu chuyển đổi o-voxel thất bại, chúng ta PHẢI dừng lại để tránh làm ô nhiễm tập dữ liệu.
                raise RuntimeError(
                    f"O-Voxel conversion failed for {obj_path}. Check if 'o_voxel' library is installed "
                    f"and CUDA is properly configured. Error: {e}"
                )

        # Logic cũ/dự phòng bị loại bỏ nghiêm ngặt để tránh làm hỏng dữ liệu.
        raise RuntimeError(f"O-Voxel logic reached unreachable code for {obj_path}")



def load_balanced_group_indices(point_counts, split_size: int):
    """Các nhóm cân bằng tải tham lam (greedy load-balancing groups) dựa trên số lượng điểm thưa thớt."""
    n = len(point_counts)
    if n <= 0:
        return []

    split = int(max(0, split_size))
    if split <= 0:
        return [list(range(n))]

    counts = [max(1, int(c)) for c in point_counts]
    total = sum(counts)
    if total <= split or n == 1:
        return [list(range(n))]

    n_groups = max(1, (total + split - 1) // split)
    groups = [[] for _ in range(n_groups)]
    group_loads = [0 for _ in range(n_groups)]

    order = sorted(range(n), key=lambda i: counts[i], reverse=True)
    for idx in order:
        g = min(range(n_groups), key=lambda k: group_loads[k])
        groups[g].append(idx)
        group_loads[g] += counts[idx]

    groups = [g for g in groups if len(g) > 0]
    return groups


def collate_voxels(batch, split_points: int = 0, spatial_size: int = 256, max_points_per_batch: int = 0):
    """
    Tùy chỉnh đối chiếu (collate) cho các mục thưa thớt có độ dài thay đổi.
    Tối ưu hóa cho RTX 4090: Thực hiện nối (concatenation) ở phía CPU VÀ giới hạn điểm (point capping) 
    trong các tiến trình worker để tránh tắc nghẽn luồng chính và đói GPU (GPU starvation).
    """
    max_pts = int(max_points_per_batch)
    
    if int(max(0, split_points)) > 0:
        # Luồng đóng gói micro-batch cân bằng (Cũ/Phức tạp)
        # Chúng ta không giới hạn (cap) ở đây vì nó được xử lý bởi logic micro-batch sau đó
        point_counts = []
        for item in batch:
            try:
                point_counts.append(int(item["features"].shape[0]))
            except Exception:
                point_counts.append(1)

        groups = load_balanced_group_indices(point_counts, int(split_points))
        if len(groups) <= 1:
            return batch

        packs = []
        for g in groups:
            packs.append([batch[i] for i in g])
        return packs

    # Luồng hiệu suất cao: Nối sẵn (Pre-concatenate) trong các tiến trình DataLoader workers
    
    # 1. Bước một: Giới hạn (Capping) nếu cần
    current_total = sum(int(item["features"].shape[0]) for item in batch)
    if max_pts > 0 and current_total > max_pts:
        ratio = max_pts / float(max(current_total, 1))
        for item in batch:
            n = int(item["features"].shape[0])
            keep = max(1, int(n * ratio))
            if keep < n:
                # Use fast deterministic sampling
                stride = n / float(keep)
                idx = (torch.arange(keep, dtype=torch.float32) * stride).to(torch.long).clamp(0, n - 1)
                item["features"] = item["features"][idx]
                if item.get("coords", None) is not None:
                    item["coords"] = item["coords"][idx]

    # 2. Bước hai: Nối (Concatenation)
    feats_list = []
    coords_list = []
    
    for b_idx, item in enumerate(batch):
        feats = item["features"]
        if feats.dtype != torch.float32:
            feats = feats.to(torch.float32)
        
        coords = item.get("coords", None)
        if coords is None:
            xyz = feats[:, :3].clamp(-1.0, 1.0)
            coords = ((xyz + 1.0) * 0.5 * (spatial_size - 1)).to(torch.int32)
        else:
            if coords.dtype != torch.int32:
                coords = coords.to(torch.int32)
        
        coords = torch.clamp(coords, 0, spatial_size - 1)
        batch_col = torch.full((coords.shape[0], 1), b_idx, dtype=torch.int32)
        coords_b = torch.cat([batch_col, coords], dim=1)
        
        feats_list.append(feats)
        coords_list.append(coords_b)

    feats_cat = torch.cat(feats_list, dim=0).contiguous()
    coords_cat = torch.cat(coords_list, dim=0).contiguous()
    
    # QUAN TRỌNG: KHÔNG ĐƯỢC bao gồm raw_items (danh sách các tensor-dicts) ở đây.
    # pin_memory=True khiến PyTorch đệ quy ghim (pin) TẤT CẢ tensor trong
    # dict trả về. Với 18 items × 2 tensors = 36 lệnh gọi page-lock kernel dư thừa
    # (~114ms bị chặn mỗi lô - batch), gây ra tình trạng đói GPU liên tục.
    # Thay vào đó, chỉ lưu trữ point_counts (số nguyên thông thường, chi phí ghim bằng không) để chúng ta
    # có thể tách feats_cat/coords_cat cho loss kết xuất stage2 ở epoch 50+.
    point_counts = [int(f.shape[0]) for f in feats_list]

    return {
        "is_pre_concatenated": True,
        "feats_cat": feats_cat,
        "coords_cat": coords_cat,
        "batch_size": len(batch),
        "point_counts": point_counts,
    }


def is_packed_micro_batch(batch_payload) -> bool:
    return (
        isinstance(batch_payload, list)
        and len(batch_payload) > 0
        and isinstance(batch_payload[0], list)
    )


def materialize_batch_items(batch_payload):
    """Chuẩn hóa các đầu ra collate thành danh sách các mục thưa thớt cho mỗi mẫu."""
    if isinstance(batch_payload, list):
        return batch_payload

    if not (isinstance(batch_payload, dict) and batch_payload.get("is_pre_concatenated")):
        return []

    feats_cat = batch_payload.get("feats_cat", None)
    coords_cat = batch_payload.get("coords_cat", None)
    point_counts = batch_payload.get("point_counts", None)
    if not isinstance(feats_cat, torch.Tensor) or feats_cat.ndim != 2:
        return []

    total_rows = int(feats_cat.shape[0])
    if total_rows <= 0:
        return []

    if not isinstance(point_counts, (list, tuple)) or len(point_counts) == 0:
        point_counts = [total_rows]

    coords_ok = (
        isinstance(coords_cat, torch.Tensor)
        and coords_cat.ndim == 2
        and int(coords_cat.shape[0]) >= total_rows
    )

    items = []
    offset = 0
    for raw_count in point_counts:
        count = int(max(0, raw_count))
        if count <= 0:
            continue

        end = min(offset + count, total_rows)
        if end <= offset:
            break

        coords = None
        if coords_ok:
            coords = coords_cat[offset:end]
            if coords.shape[1] >= 4:
                coords = coords[:, -3:]
            coords = coords.contiguous()

        items.append({
            "features": feats_cat[offset:end].contiguous(),
            "coords": coords,
        })
        offset = end

    if offset < total_rows:
        coords = None
        if coords_ok:
            coords = coords_cat[offset:total_rows]
            if coords.shape[1] >= 4:
                coords = coords[:, -3:]
            coords = coords.contiguous()
        items.append({
            "features": feats_cat[offset:total_rows].contiguous(),
            "coords": coords,
        })

    return items


def build_sparse_batch(batch_payload, device: torch.device, spatial_size: int):
    """
    Xây dựng một spconv SparseConvTensor.
    Xử lý cả định dạng danh sách từ điển (list-of-dicts) cũ và định dạng nối sẵn (pre-concatenated) được tối ưu hóa.
    """
    if spconv is None:
        raise RuntimeError("spconv is required to build sparse batch.")

    # 1. Xử lý định dạng nối sẵn được tối ưu hóa (Hiệu suất cao)
    if isinstance(batch_payload, dict) and batch_payload.get("is_pre_concatenated"):
        feats_cpu = batch_payload["feats_cat"]
        coords_cpu = batch_payload["coords_cat"]
        batch_size = batch_payload["batch_size"]
        
        if device.type == "cuda":
            # Bỏ .pin_memory() vì DataLoader đã tự động làm việc này ở background
            feats_cat = feats_cpu.to(device=device, non_blocking=True)
            coords_cat = coords_cpu.to(device=device, non_blocking=True)
        else:
            feats_cat = feats_cpu.to(device=device)
            coords_cat = coords_cpu.to(device=device)
            
        sparse = spconv.SparseConvTensor(
            features=feats_cat,
            indices=coords_cat,
            spatial_shape=[spatial_size, spatial_size, spatial_size],
            batch_size=batch_size,
        )
        return sparse, feats_cat

    # 2. Xử lý định dạng danh sách từ điển (list-of-dicts) cũ (Dự phòng/Micro-batches)
    batch_items = batch_payload
    if len(batch_items) == 0:
        raise RuntimeError("Empty sparse batch: no items provided")

    feats_all_cpu = []
    coords_all_cpu = []

    for b_idx, item in enumerate(batch_items):
        feats = item["features"]
        if feats.dtype != torch.float32:
            feats = feats.to(dtype=torch.float32)
        
        coords = item.get("coords", None)
        if coords is None:
            xyz = feats[:, :3].clamp(-1.0, 1.0)
            coords = ((xyz + 1.0) * 0.5 * (spatial_size - 1)).to(torch.int32)
        else:
            if coords.dtype != torch.int32:
                coords = coords.to(dtype=torch.int32)

        coords = torch.clamp(coords, 0, spatial_size - 1)
        batch_col = torch.full((coords.shape[0], 1), b_idx, dtype=torch.int32)
        coords_b = torch.cat([batch_col, coords], dim=1)

        feats_all_cpu.append(feats)
        coords_all_cpu.append(coords_b)

    feats_cat_cpu = torch.cat(feats_all_cpu, dim=0).contiguous()
    coords_cat_cpu = torch.cat(coords_all_cpu, dim=0).contiguous()

    if device.type == "cuda":
        # Bỏ .pin_memory() vì DataLoader đã tự động làm việc này ở background
        feats_cat = feats_cat_cpu.to(device=device, non_blocking=True)
        coords_cat = coords_cat_cpu.to(device=device, non_blocking=True)
    else:
        feats_cat = feats_cat_cpu.to(device=device)
        coords_cat = coords_cat_cpu.to(device=device)
        
    sparse = spconv.SparseConvTensor(
        features=feats_cat,
        indices=coords_cat,
        spatial_shape=[spatial_size, spatial_size, spatial_size],
        batch_size=len(batch_items),
    )
    return sparse, feats_cat


def cap_points_per_batch(batch_items, max_points_per_batch: int):
    """Giới hạn tổng số điểm thưa thớt trong một lô (batch) để làm mịn các đỉnh sử dụng VRAM."""
    batch_items = materialize_batch_items(batch_items)
    if len(batch_items) == 0:
        return [], 0, 0

    max_points = int(max_points_per_batch)
    if max_points <= 0:
        total = sum(int(item["features"].shape[0]) for item in batch_items)
        return batch_items, total, total

    total_before = sum(int(item["features"].shape[0]) for item in batch_items)
    if total_before <= max_points:
        return batch_items, total_before, total_before

    def _sample_keep_indices(n: int, keep: int) -> torch.Tensor:
        if keep >= n:
            return torch.arange(n, dtype=torch.long)
        if keep <= 1:
            return torch.tensor([random.randrange(n)], dtype=torch.long)

        keep_ratio = keep / max(float(n), 1.0)
        if keep_ratio >= 0.98:
            return torch.randperm(n)[:keep]

        stride = n / float(keep)
        offset = random.random() * stride
        idx = (offset + torch.arange(keep, dtype=torch.float32) * stride).to(torch.long)
        return idx.clamp_(0, n - 1)

    ratio = max_points / max(float(total_before), 1.0)
    capped_items = []
    for item in batch_items:
        feats = item["features"]
        coords = item.get("coords", None)
        n = int(feats.shape[0])
        keep = max(1, min(n, int(n * ratio)))
        if keep < n:
            idx = _sample_keep_indices(n, keep)
            feats = feats[idx]
            if coords is not None:
                coords = coords[idx]
        capped_items.append({
            "features": feats,
            "coords": coords,
        })

    total_after = sum(int(item["features"].shape[0]) for item in capped_items)
    overflow = total_after - max_points
    if overflow > 0:
        for item in sorted(capped_items, key=lambda x: int(x["features"].shape[0]), reverse=True):
            if overflow <= 0:
                break
            n = int(item["features"].shape[0])
            if n <= 1:
                continue
            drop = min(overflow, n - 1)
            keep = n - drop
            idx = _sample_keep_indices(n, keep)
            item["features"] = item["features"][idx]
            if item.get("coords", None) is not None:
                item["coords"] = item["coords"][idx]
            overflow -= drop

    total_after = sum(int(item["features"].shape[0]) for item in capped_items)
    return capped_items, total_before, total_after
