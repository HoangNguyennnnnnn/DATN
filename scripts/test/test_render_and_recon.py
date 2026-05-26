import argparse
import os
import sys
import torch
import torchvision
import lmdb
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.mesh_renderer import MeshRenderer
from src.data.image_preprocessor import ImagePreprocessor
from src.inference.generator import FaceDiffGenerator

def save_tensor_image(tensor, path):
    # tensor: [1, 3, H, W] in [0, 1]
    img = tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0
    img = img.astype(np.uint8)
    Image.fromarray(img).save(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True, help="Path to raw dataset .obj file")
    parser.add_argument("--id", type=str, default=None, help="LMDB ID to compare against (e.g. faceverse/064_03/064_03.obj)")
    parser.add_argument("--omega", type=float, default=4.0, help="CFG scale")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dir = "outputs_pipeline_test"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"--- 1. Rendering 3D Mesh to Front and Back Images ---")
    renderer = MeshRenderer(device=device, image_size=512)
    front_tensor, back_tensor = renderer.render_front_and_back(args.obj)
    
    front_path = os.path.join(out_dir, "temp_front.jpg")
    back_path = os.path.join(out_dir, "temp_back.jpg")
    
    save_tensor_image(front_tensor, front_path)
    save_tensor_image(back_tensor, back_path)
    print(f"Saved rendered images to {front_path} and {back_path}")
    
    print(f"--- 2. Extracting Context from Rendered Images ---")
    preprocessor = ImagePreprocessor(device=device)
    image_context = preprocessor.process(front_path, back_path)
    image_context = image_context.to(device).float()
    print(f"Image context shape: {image_context.shape}")
    
    if args.id:
        print(f"--- 3. Fetching Ground-Truth Context from LMDB ---")
        env = lmdb.open("data/hybrid_context.lmdb", readonly=True, lock=False)
        import io
        with env.begin() as txn:
            val = txn.get(args.id.encode())
            if val is not None:
                lmdb_context = torch.load(io.BytesIO(val), weights_only=False).to(device).float()
                mse = torch.nn.functional.mse_loss(image_context, lmdb_context)
                print(f"--- [Analysis] MSE between GT LMDB context and Rendered Image context: {mse.item():.6f} ---")
            else:
                print(f"WARNING: ID {args.id} not found in LMDB!")
                lmdb_context = None
    
    print("--- 4. Generating Mesh from Rendered Image Context ---")
    generator = FaceDiffGenerator(
        device=device,
        imf_ckpt="checkpoints/imf_v8_lite/epoch_100.pt",
        sc_vae_shape_ckpt="checkpoints/sc_vae_shape/epoch_500.pt",
        cfg_scale=args.omega,
        feature_mode="poisson",
        enforce_dual_contouring=False,
    )
    
    basename = os.path.basename(args.obj).split('.')[0]
    out_img = os.path.join(out_dir, f"{basename}_recon_from_rendered.ply")
    generator.generate(
        context=image_context.unsqueeze(0),
        output_path=out_img,
        omega=args.omega
    )
    print(f"Saved Reconstructed Mesh: {out_img}")
    
    if args.id and lmdb_context is not None:
        out_lmdb = os.path.join(out_dir, f"{basename}_recon_from_gt_lmdb.ply")
        generator.generate(
            context=lmdb_context.unsqueeze(0),
            output_path=out_lmdb,
            omega=args.omega
        )
        print(f"Saved Ground-Truth LMDB Mesh: {out_lmdb}")
        
    print("Done!")

if __name__ == "__main__":
    main()
