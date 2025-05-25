import os
import sys
from multiprocessing import Pool, cpu_count
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
import torch
from PIL import Image

# Global variables for multiprocessing
predictor = None
checkpoint_path = None
model_cfg_path = None

def init_worker(checkpoint, model_cfg):
    """Initialize the predictor in each worker process"""
    global predictor
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def process_single_image(args):
    input_path, output_path = args
    
    # depth_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    depth_image = Image.open(input_path)
    depth_image = np.array(depth_image.convert('RGB') )
    
    if depth_image is None:
        print(f"[ERROR] Failed to read image: {input_path}")
        return
    
    print(depth_image.shape)
    # print(depth_image.dtype)
    
    depth_image = depth_image.astype(np.float32) / 10000.0
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Set the image for SAM2 predictor
        predictor.set_image(depth_image)
        
        # Compute SAM2 mask - you need to provide input prompts
        # This is a placeholder - you'll need to add proper prompts
        # For example, using a point prompt:
        input_point = np.array([[depth_image.shape[1]//2, depth_image.shape[0]//2]])  # center point
        input_label = np.array([1])  # foreground
        
        mask, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
    
        # Save image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Convert boolean mask to uint8
        mask_uint8 = (mask[0] * 255).astype(np.uint8)
        cv2.imwrite(output_path, mask_uint8)
        print(f"Processed {input_path} to {output_path}")

def collect_image_paths(input_dir, output_dir):
    image_pairs = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.endswith('.png'):
                continue
            input_path = os.path.join(root, filename)
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)

            # Skip if we've already generated this HHA to save compute
            if os.path.exists(output_path):
                # print(f"Skipping {output_path}. Image already exists")
                continue
            # print(f"adding {input_path} to {output_path}")
            image_pairs.append((input_path, output_path))
    return image_pairs

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_sam2mask.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    image_paths = collect_image_paths(input_directory, output_directory)
    if not image_paths:
        print("No new images to process")
        sys.exit(0)
    
    checkpoint = "//home/kotaik/DH/EgoDepth-HMR/thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "//home/kotaik/DH/EgoDepth-HMR/thirdparty/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # num_workers = min(cpu_count(), len(image_paths))
    num_workers = 5
    print(f"Using {num_workers} workers for processing {len(image_paths)} images.")
    
    # Use initializer to set up predictor in each worker process
    with Pool(num_workers, initializer=init_worker, initargs=(checkpoint, model_cfg)) as pool:
        pool.map(process_single_image, image_paths)
