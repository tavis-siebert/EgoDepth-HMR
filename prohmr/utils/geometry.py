from typing import Optional
import torch
from torch.nn import functional as F
from typing import Tuple
import numpy as np


def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def rot6d_to_rotmat(x, rot6d_mode='prohmr'):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    if rot6d_mode == 'prohmr':
        x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    elif rot6d_mode == 'diffusion':
        x = x.reshape(-1, 3, 2)
    ### note: order for 6d feture items different between diffusion and prohmr code!!!
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(x_batch, rot6d_mode='prohmr'):
    # x_batch: [:,3,3]
    if rot6d_mode == 'diffusion':
        xr_repr = x_batch[:, :, :-1].reshape([-1, 6])
    else:
        pass  # todo
    return xr_repr


def perspective_projection(points: torch.Tensor,
                           translation: torch.Tensor,
                           focal_length: torch.Tensor,
                           camera_center: Optional[torch.Tensor] = None,
                           rotation: Optional[torch.Tensor] = None,
                           return_depths = False) -> torch.Tensor:
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    # print("points shape after rotation: ", points.size())
    # print("translation shape: ", translation.size())
    points = points + translation.unsqueeze(1)
    
    depths = points[:, :, -1].unsqueeze(-1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    
    if return_depths:
        return projected_points[:, :, :-1], depths
    else:
        return projected_points[:, :, :-1]
    
    
def render_keypoints_to_depth_map_fast(
        keypoints_2d: torch.Tensor,
        keypoints_depth: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rasterizes sparse keypoint depths into a depth map using scatter-reduce.
        
        Args:
            keypoints_2d: (B, N, 2) projected 2D pixel coordinates
            keypoints_depth: (B, N) depth values
            image_size: (H, W)
        
        Returns:
            depth_map: (B, H, W)
            valid_mask: (B, H, W)
        """
        # print("keypoints_2d shape: ", keypoints_2d.shape)
        # print("keypoints_depth shape: ", keypoints_depth.shape)
        
        B = keypoints_depth.size(0)
        H, W = image_size
        device = keypoints_depth.device

        # Round and clamp 2D pixel coordinates
        x = keypoints_2d[..., 0].round().long().clamp(0, W - 1)
        y = keypoints_2d[..., 1].round().long().clamp(0, H - 1)

        # Compute linear pixel indices
        linear_idx = y * W + x  # Shape: (B, N)

        # Flatten everything
        flat_idx = linear_idx.view(-1)
        flat_depth = keypoints_depth.flatten()

        # Prepare output tensor
        depth_flat = torch.full((B * H * W,), float('inf'), device=device)

        # Create batch offset to index into [0, B*H*W)
        batch_offsets = (
            torch.arange(B, device=device).unsqueeze(1) * (H * W)
        )  # Shape: (B, 1)
        flat_idx_with_batch = (linear_idx + batch_offsets).view(-1)

        # Do scatter_reduce to perform z-buffer min
        depth_flat = torch.scatter_reduce(
            input=depth_flat,
            dim=0,
            index=flat_idx_with_batch,
            src=flat_depth,
            reduce="amin",
            include_self=True
        )

        # Reshape and compute valid mask
        depth_map = depth_flat.view(B, H, W)
        valid_mask = depth_map != float('inf')

        return depth_map, valid_mask
    
    
def project_on_depth(points, intrinsic_matrix, width, height):
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    xy, _ = cv2.projectPoints(points, rvec, tvec, intrinsic_matrix, None)
    xy = np.squeeze(xy)
    xy = np.around(xy).astype(int)

    width_check = np.logical_and(0 <= xy[:, 0], xy[:, 0] < width)
    height_check = np.logical_and(0 <= xy[:, 1], xy[:, 1] < height)
    valid_ids = np.where(np.logical_and(width_check, height_check))[0]
    xy = xy[valid_ids, :]

    z = points[valid_ids, 2]
    depth_image = np.zeros((height, width))
    rgb = rgb[valid_ids, :]
    rgb = rgb[:, ::-1]
    for i, p in enumerate(xy):
        depth_image[p[1], p[0]] = z[i]

    return depth_image

import torch

def project_on_depth_torch_batch(points: torch.Tensor, intrinsic_matrix: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    Projects a batch of 3D point clouds into depth images.

    Args:
        points: (B, S, N, 3) 3D points in camera space (S = num_samples).
        intrinsic_matrix: (B, 3, 3) or (1, 3, 3) camera intrinsics.
        width: image width.
        height: image height.

    Returns:
        depth_images: (B, S, 1, H, W) depth images.
    """
    assert points.ndim == 4 and points.shape[-1] == 3, "points must be of shape (B, S, N, 3)"
    B, S, N, _ = points.shape
    device = points.device
    dtype = points.dtype

    # Expand intrinsics if shared
    if intrinsic_matrix.shape == (3, 3):
        intrinsic_matrix = intrinsic_matrix.unsqueeze(0).expand(B, -1, -1)
    assert intrinsic_matrix.shape == (B, 3, 3), "intrinsic_matrix must be of shape (B, 3, 3)"

    fx = intrinsic_matrix[:, 0, 0].view(B, 1, 1)  # (B,1,1)
    fy = intrinsic_matrix[:, 1, 1].view(B, 1, 1)
    cx = intrinsic_matrix[:, 0, 2].view(B, 1, 1)
    cy = intrinsic_matrix[:, 1, 2].view(B, 1, 1)

    X = points[..., 0]  # (B, S, N)
    Y = points[..., 1]
    Z = points[..., 2]

    # Pixel projection
    x = (X * fx / Z + cx).round().long()
    y = (Y * fy / Z + cy).round().long()

    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (Z > 0)

    # Output tensor
    depth_images = torch.zeros((B, S, 1, height, width), device=device, dtype=dtype)

    for b in range(B):
        for s in range(S):
            valid = valid_mask[b, s]
            x_b = x[b, s][valid]
            y_b = y[b, s][valid]
            z_b = Z[b, s][valid]
            depth_images[b, s, 0, y_b, x_b] = z_b

    return depth_images


def center_crop_batch(images: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    """
    Center-crop a batch of images.

    Args:
        images: (B, C, H, W) input tensor.
        crop_h: desired crop height.
        crop_w: desired crop width.

    Returns:
        (B, C, crop_h, crop_w) cropped tensor.
    """
    B, _, h, w = images.shape
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return images[:, :, top:top + crop_h, left:left + crop_w]

def crop_around_bbox_center(images, bbox_center, crop_size=224):
    """
    Crop 224×224 patches around bbox_center from full-size images.

    Args:
        images: (B, 1, H, W)
        bbox_center: (B, 2) in full image coordinates (x, y)
        crop_size: int

    Returns:
        cropped_images: (B, 1, crop_size, crop_size)
    """
    B, C, H, W = images.shape
    cropped = torch.zeros((B, C, crop_size, crop_size), device=images.device)

    for i in range(B):
        cx, cy = int(bbox_center[i, 0]), int(bbox_center[i, 1])
        x1 = max(cx - crop_size // 2, 0)
        y1 = max(cy - crop_size // 2, 0)
        x2 = min(x1 + crop_size, W)
        y2 = min(y1 + crop_size, H)
        cropped[i, :, :y2 - y1, :x2 - x1] = images[i, :, y1:y2, x1:x2]

    return cropped

def backproject_2d_to_3d(
    masked_coords: torch.Tensor,
    depth_map: torch.Tensor,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Backprojects N 2D points in a depth map into 3D points in camera space.
    
    Args:
        masked_coords: (N, 2) 2D pixel coordinates of points to backproject.
        depth_map: (H, W) depth map where each pixel contains the depth value in meters.
        device: torch device to use for computation.
        
    Returns:
        points_3d: (B, H*W, 3) 3D points in camera space.
    """
    scale = 1
    width = 320 * scale
    height = 288 * scale
    focal_length = 200 * scale
    intrinsic_matrix = torch.tensor([[focal_length, 0, width // 2],
                            [0, focal_length, height // 2],
                            [0, 0, 1.]])
    
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    us = masked_coords[:, 1].float()
    vs = masked_coords[:, 0].float()
    ds = depth_map[vs.long(), us.long()] * 5  # Scale back to meters

    X = (us - cx) * ds / fx
    Y = (vs - cy) * ds / fy
    Z = ds
    backproj_points = torch.stack([X, Y, Z], dim=-1).to(device)  # [N, 3]
    
    return backproj_points
