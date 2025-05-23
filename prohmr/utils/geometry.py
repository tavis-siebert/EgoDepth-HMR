from typing import Optional
import torch
from torch.nn import functional as F
from typing import Tuple


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


def depth_to_3dpointcloud(depth_map: torch.Tensor,
                          translation: torch.Tensor,
                          focal_length: torch.Tensor,
                          camera_center: Optional[torch.Tensor] = None,
                          rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Converts a depth map to a 3D point cloud.
    Args:
        depth_map (torch.Tensor): Tensor of shape (B, H, W) containing the depth map.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 3) containing the 3D point cloud.
    """
    batch_size, height, width = depth_map.shape
    if rotation is None:
        rotation = torch.eye(3, device=depth_map.device, dtype=depth_map.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=depth_map.device, dtype=depth_map.dtype)

    u = torch.arange(width, device=depth_map.device)
    v = torch.arange(height, device=depth_map.device)
    u, v = torch.meshgrid(u, v, indexing='xy')  # (H, W)

    u = u.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H, W)
    v = v.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H, W)

    fx = focal_length[:, 0].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    fy = focal_length[:, 1].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    cx = camera_center[:, 0].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    cy = camera_center[:, 1].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    fx = fx.to(depth_map.device)
    fy = fy.to(depth_map.device)
    cx = cx.to(depth_map.device)
    cy = cy.to(depth_map.device)

    # 3D coordinates
    x = (u - cx) * depth_map / fx  # (B, H, W)
    y = (v - cy) * depth_map / fy  # (B, H, W)
    z = depth_map  # (B, H, W) # TODO: depth scale?
    points_3d_cam = torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)
    points_3d_cam = points_3d_cam.reshape(batch_size, height * width, 3) # (B, H*W, 3)
    
    # Apply rotation: points_world = R @ points_cam + t
    # points_3d_cam: (B, N, 3), rotation: (B, 3, 3)
    points_3d_world = torch.bmm(points_3d_cam, rotation.transpose(-2, -1))  # (B, H*W, 3)
    
    # Add translation
    translation_expanded = translation.unsqueeze(1)  # (B, 1, 3)
    points_3d_world = points_3d_world + translation_expanded  # (B, H*W, 3)
    
    valid_mask = depth_map.reshape(batch_size, -1) > 0  # (B, H*W)
    
    # Create output tensor with maximum possible points
    max_points = height * width
    output_points = torch.zeros(batch_size, max_points, 3, device=depth_map.device, dtype=depth_map.dtype)
    
    # Fill valid points for each batch item
    for b in range(batch_size):
        valid_indices = valid_mask[b]
        valid_points = points_3d_world[b][valid_indices]  # (num_valid, 3)
        num_valid = valid_points.shape[0]
        output_points[b, :num_valid] = valid_points
    
    return output_points # (B, H*W, 3)


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
    
    
    
