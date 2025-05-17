import os
import sys
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np

from thirdparty.Depth2HHA.getHHA import getHHA

CAMERA_MATRIX = np.array([[200., 0., 160.],
                              [0., 200., 144.],
                              [0., 0., 1.]])

def process_single_image(args):
    input_path, output_path = args

    # Read depth image
    depth_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print(f"[ERROR] Failed to read image: {input_path}")
        return
    depth_image = depth_image.astype(np.float32) / 10000.0

    # Compute HHA
    hha_image = getHHA(CAMERA_MATRIX, depth_image, depth_image)

    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, hha_image)
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

            # # Skip if we've already generated this HHA to save compute
            # if os.path.exists(output_path):
            #     print(f"Skipping {output_path}. Image already exists")
            #     continue
            # print(f"adding {input_path} to {output_path}")
            image_pairs.append((input_path, output_path))

    return image_pairs

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_hha.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    image_paths = collect_image_paths(input_directory, output_directory)
    if not image_paths:
        print("No new images to process")
        sys.exit(0)

    num_workers = min(cpu_count(), len(image_paths))
    with Pool(num_workers) as pool:
        pool.map(process_single_image, image_paths)
