import numpy as np
import cv2
import torch
import torch.nn.functional as F

def compute_normals_from_depth(depth, fx=200., fy=200., cx=0., cy=0.):
    # print("depth shape: ", depth.shape)
    # depth = depth[:, :, 0]
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))

    # Convert pixel coordinates to camera coordinates
    Z = depth
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    # Stack into 3D point cloud
    points = np.stack((X, Y, Z), axis=2)  # shape: (H, W, 3)

    # Compute gradients
    dzdx = np.gradient(points, axis=1)
    dzdy = np.gradient(points, axis=0)

    # Cross product of gradients to get normal
    normals = np.cross(dzdx, dzdy)
    
    # Normalize
    norm = np.linalg.norm(normals, axis=2, keepdims=True) + 1e-8
    normals_normalized = normals / norm
    

    return torch.tensor(normals_normalized).permute(2, 0, 1).float()  # shape: (H, W, 3), values in [-1, 1]

def compute_normals_from_depth_batch(depth, fx=200., fy=200., cx=0., cy=0.):
    """
    Compute surface normals from a batch of depth maps using central differences.

    Args:
        depth (torch.Tensor): (B, H, W) depth maps in meters
        fx, fy (float): focal lengths
        cx, cy (float): principal point

    Returns:
        normals (torch.Tensor): (B, H, W, 3) surface normals, unit vectors in [-1, 1]
    """
    B, H, W = depth.shape
    device = depth.device

    # Create meshgrid once
    x = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)
    y = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)

    Z = depth
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    # Stack to point cloud: (B, H, W, 3)
    points = torch.stack((X, Y, Z), dim=-1)

     # Use central differences (torch.roll ensures consistent shape)
    points_x1 = torch.roll(points, shifts=-1, dims=2)
    points_x2 = torch.roll(points, shifts=1, dims=2)
    dx = (points_x1 - points_x2) / 2.0

    points_y1 = torch.roll(points, shifts=-1, dims=1)
    points_y2 = torch.roll(points, shifts=1, dims=1)
    dy = (points_y1 - points_y2) / 2.0
    
    # Cross product to get normals
    normals = torch.cross(dx, dy, dim=-1)

    # Normalize
    normals = F.normalize(normals, dim=-1)
    
    normals = ((normals + 1.0) * 0.5 * 255.0)

    return normals  # (B, H, W, 3)

def compute_normals_simple(depth):
    """
    Approximates surface normals from a depth image using central differences.
    Equivalent to the C++ OpenCV implementation provided.
    
    Args:
        depth (np.ndarray): Input depth image of shape (H, W), dtype float32.

    Returns:
        normals (np.ndarray): Output normal map of shape (H, W, 3), dtype float32.
    """
    depth = depth[:, :, 0]
    H, W = depth.shape
    normals = np.zeros((H, W, 3), dtype=np.float32)

    # Pad to handle borders (use edge padding to avoid NaNs)
    padded = np.pad(depth, ((1, 1), (1, 1)), mode='edge')

    # Compute gradients using central difference
    dzdx = (padded[2:, 1:-1] - padded[0:-2, 1:-1]) / 2.0
    dzdy = (padded[1:-1, 2:] - padded[1:-1, 0:-2]) / 2.0

    # Construct normal vectors
    normals[..., 0] = -dzdx
    normals[..., 1] = -dzdy
    normals[..., 2] = 1.0

    # Normalize
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= (norm + 1e-8)

    return normals

def visualize_normals(normals):
    # Convert [-1, 1] to [0, 255]
    normals_vis = ((normals + 1.0) * 0.5 * 255.0).astype(np.uint8)
    return normals_vis


if __name__ == "__main__":
    # Example usage
    depth = cv2.imread("/home/weiwan/DigitalHuman/EgoDepth-HMR/output/failed_img.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # in meters
    fx, fy, cx, cy = 200.0, 200.0, 0., 0.  # example intrinsics

    normals = compute_normals_from_depth(depth, fx, fy, cx, cy)
    # normals = compute_normals_simple(depth)
    normal_img = visualize_normals(normals)

    cv2.imwrite("/home/weiwan/DigitalHuman/EgoDepth-HMR/output/normal_map.png", normal_img)
