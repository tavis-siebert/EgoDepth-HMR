import os
import cv2
from thirdparty.Depth2HHA.getHHA import getHHA  # remove the dot if it's a top-level import
import numpy as np
import sys

def process_depth_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.png'):
                input_path = os.path.join(root, filename)
                # Maintain relative subdirectory structure
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                depth_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                if depth_image is None:
                    print(f"Failed to read image: {input_path}")
                    continue
                depth_image = depth_image.astype(np.float32) / 10000.0

                # Camera intrinsics (you can adjust these as needed)
                camera_matrix = np.array([[200., 0., 160.],
                                          [0., 200., 144.],
                                          [0., 0., 1.]])

                # Generate HHA image
                hha_image = getHHA(camera_matrix, depth_image, depth_image)

                # Save the image
                cv2.imwrite(output_path, hha_image)
                print(f"Processed {input_path} to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_hha.py <input_directory> <output_directory>")
        sys.exit(1)
        
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    # Process the depth images
    process_depth_images(input_directory, output_directory)
