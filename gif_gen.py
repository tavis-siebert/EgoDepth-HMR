from PIL import Image
import os

def generate_gif_from_images(image_folder, output_path, duration=200, loop=0):
    """
    Generate a GIF from a set of images in a folder.

    Args:
        image_folder (str): Path to the folder containing images.
        output_path (str): Path where the output GIF will be saved.
        duration (int): Time (ms) to display each frame.
        loop (int): Number of loops (0 = infinite).
    """
    # Get all image files and sort them
    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        print("No image files found in the folder.")
        return

    # Open images
    frames = [Image.open(img) for img in image_files]

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )
    print(f"GIF saved to {output_path}")

# Example usage
if __name__ == "__main__":
    generate_gif_from_images(
        image_folder="output/pred_depth",          # Change this to your image folder
        output_path="output/pred_depth/pred_depth.gif",       # Name of your output GIF
        duration=600,                   # Duration per frame in ms
        loop=0                          # 0 = loop forever
    )
