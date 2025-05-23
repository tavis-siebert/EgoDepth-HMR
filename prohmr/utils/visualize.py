import matplotlib.pyplot as plt
import torch
import os

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import imageio



def save_reprojection_images(reprojected_pts, img_sz, save_dir: str):
    
    img = torch.zeros((img_sz[0], img_sz[1]), dtype=torch.uint8)
    num_images = min(10, reprojected_pts.shape[0])  # Limit to first 10 images
    os.makedirs(save_dir, exist_ok=True)
    for i in range(num_images):
        img.fill_(0)  # Clear the image
        pts = reprojected_pts[i].cpu().numpy()
        pts = pts[:, :].astype(int)  # Convert to integer pixel coordinates
        img[pts[:, 1], pts[:, 0]] = 255  # Set pixel value to white

        save_path = os.path.join(save_dir, f"reprojected_{i:02d}.png")
        plt.imsave(save_path, img.numpy(), cmap='gray')


def save_depth_image(gt_depth_imgs: torch.Tensor, depth_map: torch.Tensor, save_dir: str):
    """
    Saves the first 10 depth images in a batch as grayscale visualizations.

    Args:
        depth_map (torch.Tensor): (B, H, W) depth tensor.
        save_dir (str): Output directory to save images.
    """
    assert depth_map.dim() == 3, "Expected shape (B, H, W)"
    print("shape of depth_map: ", depth_map.shape)
    print("shape of gt_depth_imgs: ", gt_depth_imgs.shape)
    os.makedirs(save_dir, exist_ok=True)

    num_images = min(10, depth_map.size(0))  # Limit to first 10 images

    for i in range(num_images):
        depth = depth_map[i].detach().cpu()
        gt_depth = gt_depth_imgs[i].detach().cpu()  

        # Handle invalid values (e.g., inf)
        mask = ~torch.isinf(depth)
        valid_depth = depth[mask]
        if valid_depth.numel() == 0:
            print(f"[Warning] Image {i} has no valid depth.")
            continue

        valid_depth = valid_depth.numpy()
        depth = depth.numpy()

        # Normalize depth to [0, 1] for visualization
        min_val, max_val = valid_depth.min(), valid_depth.max()
        norm_depth = (depth - min_val) / (max_val - min_val + 1e-8)
        norm_depth[~mask.numpy()] = 0  # Optional: make invalid pixels black

        save_path = os.path.join(save_dir, f"depth_{i:02d}.png")
        gt_save_path = os.path.join(save_dir, f"gt_depth_{i:02d}.png")
        plt.imsave(gt_save_path, gt_depth, cmap='gray')
        plt.imsave(save_path, norm_depth, cmap='gray')
        

def render_point_cloud_gif(point_cloud: torch.Tensor, output_path='point_cloud.gif', num_views=36):
    """
    Renders multiple views of a point cloud and saves as a GIF.
    
    Args:
        point_cloud (torch.Tensor): Tensor of shape (N, 3)
        output_path (str): Path to save the GIF.
        num_views (int): Number of different angles to render.
    """
    assert point_cloud.shape[1] == 3, "Point cloud should be of shape (N, 3)"
    
    point_cloud_np = point_cloud.cpu().numpy()
    images = []
    tmp_dir = "tmp_pc_views"
    os.makedirs(tmp_dir, exist_ok=True)
    
    for i, angle in enumerate(np.linspace(0, 360, num_views, endpoint=False)):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud_np[:, 0], point_cloud_np[:, 1], point_cloud_np[:, 2],
                   c='blue', s=1)
        ax.view_init(elev=20, azim=angle)
        ax.set_axis_off()
        
        frame_path = os.path.join(tmp_dir, f"view_{i:03d}.png")
        plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        images.append(imageio.imread(frame_path))
    
    imageio.mimsave(output_path, images, fps=12)
    
    # Clean up temporary files
    for img_path in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, img_path))
    os.rmdir(tmp_dir)
    
    print(f"GIF saved to {output_path}")


        
        

