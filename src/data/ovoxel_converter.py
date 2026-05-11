import os
import sys
import torch
import numpy as np
import trimesh
from typing import Dict, Optional, Tuple, Union

# Absolute pathing for o_voxel
ovoxel_path = "/mnt/18TData/facediff/third_party/TRELLIS.2/o-voxel"
if ovoxel_path not in sys.path:
    sys.path.append(ovoxel_path)

try:
    import o_voxel
    from o_voxel.convert.flexible_dual_grid import mesh_to_flexible_dual_grid
    from o_voxel.convert.volumetic_attr import textured_mesh_to_volumetric_attr
    from o_voxel.serialize import encode_seq
except ImportError:
    o_voxel = None

class OVoxelConverter:
    """
    Microsoft Research Standard O-Voxel Converter (v13.9).
    - Perfect Precision Alignment.
    - 10-Channel Output for SC-VAE Compatibility (dv3, flag3, gamma1, rgb3).
    """
    def __init__(self, resolution: int = 256, **kwargs):
        self.resolution = int(resolution)

    def _pbr_ify(self, mesh):
        from trimesh.visual.material import PBRMaterial
        if not isinstance(mesh.visual.material, PBRMaterial):
            image = getattr(mesh.visual.material, 'image', getattr(mesh.visual.material, 'diffuse', None))
            mesh.visual.material = PBRMaterial(
                baseColorFactor=[255, 255, 255, 255],
                baseColorTexture=image,
                metallicFactor=0.0,
                roughnessFactor=1.0
            )
        return mesh

    def process_mesh(self, obj_path: str):
        if o_voxel is None:
            raise ImportError("o_voxel library not found.")

        # 1. Load & PBR-ify
        asset = trimesh.load(obj_path, process=False)
        mesh = asset.to_mesh() if isinstance(asset, trimesh.Scene) else asset
        mesh = self._pbr_ify(mesh)
        
        # 2. Normalization [-0.5, 0.5]
        v_world = torch.from_numpy(mesh.vertices).float()
        aabb_min, aabb_max = v_world.min(dim=0)[0], v_world.max(dim=0)[0]
        center = (aabb_min + aabb_max) / 2.0
        scale = 0.95 / (aabb_max - aabb_min).max().clamp_min(1e-8)
        mesh.vertices = ((v_world - center) * scale).numpy()
        norm_params = {'center': center, 'scale': scale, 'res': self.resolution}

        # 3. Geometry Voxelization
        aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)
        res_geo = mesh_to_flexible_dual_grid(
            torch.from_numpy(mesh.vertices).float(), torch.from_numpy(mesh.faces).long(),
            grid_size=self.resolution, aabb=aabb, regularization_weight=1e-2
        )
        coords_geo, dv_geo, flag_geo = res_geo['coords'], res_geo['dual_vertices'], res_geo['intersected_flag']

        # 4. Material Voxelization
        coords_mat, attrs = textured_mesh_to_volumetric_attr(mesh, grid_size=self.resolution, aabb=aabb)
        base_color = attrs['base_color'] 

        # 5. Morton Alignment
        vid_geo = encode_seq(coords_geo.cuda()).cpu()
        vid_mat = encode_seq(coords_mat.cuda()).cpu()
        order_geo = torch.argsort(vid_geo)
        coords_geo, dv_geo, flag_geo, vid_geo = coords_geo[order_geo], dv_geo[order_geo], flag_geo[order_geo], vid_geo[order_geo]
        order_mat = torch.argsort(vid_mat)
        coords_mat, base_color, vid_mat = coords_mat[order_mat], base_color[order_mat], vid_mat[order_mat]

        common_vids = np.intersect1d(vid_geo.numpy(), vid_mat.numpy())
        mask_geo = torch.from_numpy(np.isin(vid_geo.numpy(), common_vids))
        mask_mat = torch.from_numpy(np.isin(vid_mat.numpy(), common_vids))
        coords, dv, flag, color = coords_geo[mask_geo], dv_geo[mask_geo], flag_geo[mask_geo], base_color[mask_mat]

        # 6. Packing 10-CHANNELS (dv3, flag3, gamma1, rgb3)
        dv_local = (dv * self.resolution - coords.float()).clamp(0.0, 1.0)
        flag_float = flag.to(torch.float32)
        dv_var = dv_local.var(dim=1, keepdim=True)
        gamma = (1.0 - dv_var).clamp(0.0, 1.0)
        color_float = color.to(torch.float32) / 255.0

        feat = torch.cat([dv_local, flag_float, gamma, color_float], dim=-1)

        print(f"  -> Processed {len(coords)} aligned 10-ch voxels.")
        return {'coords': coords, 'shape_mat_features': feat, 'aabb': aabb, 'norm_params': norm_params, 'resolution': self.resolution}