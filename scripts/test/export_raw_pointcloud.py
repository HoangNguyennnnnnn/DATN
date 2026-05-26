import sys
import os
import torch
import numpy as np
import trimesh
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.inference.generator import FaceDiffGenerator
from src.data.image_preprocessor import ImagePreprocessor
from src.data.mesh_renderer import MeshRenderer

def export_raw_pcd(obj_path, out_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.basename(obj_path).split('.')[0]
    
    # 1. Render & Extract Context (same as before)
    print("Rendering...")
    renderer = MeshRenderer()
    front_img, back_img = renderer.render_front_and_back(obj_path)
    front_path = os.path.join(out_dir, f"{basename}_temp_front.jpg")
    back_path = os.path.join(out_dir, f"{basename}_temp_back.jpg")
    import torchvision
    front_img = torchvision.transforms.functional.to_pil_image(front_img[0])
    back_img = torchvision.transforms.functional.to_pil_image(back_img[0])
    front_img.save(front_path)
    back_img.save(back_path)
    
    preprocessor = ImagePreprocessor(device=device)
    image_context = preprocessor.process(front_path, back_path)
    
    # 2. Setup Generator
    generator = FaceDiffGenerator(
        device=device,
        imf_ckpt="checkpoints/imf_v8_lite/epoch_100.pt",
        sc_vae_ckpt="checkpoints/sc_vae_shape/epoch_500.pt"
    )
    
    # 3. Step 1: iMF Generate Slats
    print("Generating Slat Tokens...")
    context = image_context.unsqueeze(0).to(device)
    b = context.size(0)
    
    with torch.no_grad():
        # manual forward
        generated_slats = generator.imf.sample_1_step(
            generator.unet,
            context,
            shape=(b, generator.slat_length, generator.stage2_input_dim),
            omega=1.0,
            cfg_tmin=0.0,
            cfg_tmax=1.0,
        )
        
        if generator.slat_mean is not None and generator.slat_std is not None:
            d_dim = generated_slats.shape[-1]
            s_dim = generator.slat_mean.shape[-1]
            if d_dim >= s_dim:
                generated_slats[..., :s_dim] = generated_slats[..., :s_dim] * generator.slat_std + generator.slat_mean
                
        # 4. Step 2: SC-VAE Decode
        print("Decoding Slats to Voxel Features...")
        grid_indices = generator._get_slat_grid_indices(batch_size=b, grid_size=generator.slat_grid_size)
        generated_slats_2d = generated_slats.view(-1, generator.slat_dim)
        voxel_features, _, _, _ = generator.vae.decode(
            generated_slats_2d, original_indices=grid_indices, batch_size=b, return_indices=True
        )
    
    print(f"Raw Voxel Features Shape: {voxel_features.shape}")
    
    # 5. Export Raw Point Cloud (OVoxel Layout)
    feats = voxel_features.detach().cpu().float().numpy()
    out_indices = out_indices.detach().cpu().numpy()
    
    # Extract coords from indices (strip batch column)
    coords = out_indices[:, 1:4].astype(np.float32)
    
    # Extract intra-voxel offsets (v)
    v = np.clip(feats[:, :3], 0.0, 1.0)
    
    # Compute global spatial coordinates based on AABB [-1, 1] and grid_size 256
    grid_size = 256.0
    voxel_size = 2.0 / grid_size
    points = (coords + v) * voxel_size - 1.0
    
    colors = None
    if feats.shape[1] >= 10:
        colors = np.nan_to_num(feats[:, 7:10], nan=0.5, posinf=1.0, neginf=0.0)
        colors = np.clip(colors, 0.0, 1.0) * 255.0
        colors = colors.astype(np.uint8)
    
    pcd = trimesh.points.PointCloud(vertices=points, colors=colors)
    out_path = os.path.join(out_dir, f"{basename}_raw_pointcloud.ply")
    pcd.export(out_path)
    print(f"Exported raw point cloud to {out_path} ({len(points)} points)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs_pipeline_test")
    args = parser.parse_args()
    export_raw_pcd(args.obj, args.out)
