import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.image_preprocessor import ImagePreprocessor
from src.inference.generator import FaceDiffGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", type=str, required=True, help="Path to front image")
    parser.add_argument("--back", type=str, default=None, help="Path to back image (optional)")
    parser.add_argument("--omega", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--output", type=str, default="outputs_pipeline_test/real_image_mesh.ply")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("--- 1. Initializing Image Preprocessor ---")
    preprocessor = ImagePreprocessor(device=device)
    
    print(f"--- 2. Extracting Context from {args.front} ---")
    context = preprocessor.process(args.front, args.back)
    context = context.unsqueeze(0).to(device) # [1, 946]
    
    print("--- 3. Initializing FaceDiff Generator ---")
    generator = FaceDiffGenerator(
        device=device,
        imf_ckpt="checkpoints/imf_v8_lite/epoch_100.pt",
        sc_vae_shape_ckpt="checkpoints/sc_vae_shape/epoch_500.pt",
        cfg_scale=args.omega,
    )
    
    print(f"--- 4. Generating Mesh with CFG Scale (Omega) = {args.omega} ---")
    out_path = generator.generate(
        context=context,
        output_path=args.output,
        omega=args.omega
    )
    
    print(f"Success! Mesh saved to: {out_path}")

if __name__ == "__main__":
    main()
