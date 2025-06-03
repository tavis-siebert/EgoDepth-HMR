import cv2
import numpy as np
import torch
from thirdparty.pytorch_openpose.src.body import Body
from thirdparty.pytorch_openpose.src import util

body_estimation = Body('/work/courses/digital_human/13/pytorch-openpose/model/body_pose_model.pth')

def estimate_pelvis_from_surface_normals(img_tensor, depth_map, intrinsic_matrix, return_3d=True):
    """
    Estimate 3D pelvis position from surface normals using OpenPose and depth map.

    Args:
        img_tensor (torch.Tensor): (3, H, W), surface normal RGB image.
        depth_map (torch.Tensor): (H, W), depth image aligned with surface normals.
        intrinsic_matrix (torch.Tensor): (3, 3), camera intrinsics.

    Returns:
        torch.Tensor: (3,), 3D pelvis point in camera coordinates if return_3d is True.
        np.ndarray: (2,), 2D pelvis UV coordinates in the image if return_3d is False.
    """
    # Convert to OpenPose compatible format
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    candidate, subset = body_estimation(img_bgr)
    if len(subset) == 0 or subset[0][8] == -1 or subset[0][11] == -1:
        # print("No valid pelvis keypoints found in the image.")
        return None
    

    rhip = candidate[int(subset[0][8])][:2]
    lhip = candidate[int(subset[0][11])][:2]
    pelvis_uv = ((rhip + lhip) / 2.).astype(int)

    u, v = pelvis_uv[0], pelvis_uv[1]
    
    if not return_3d:
        if v >= depth_map.shape[0] or u >= depth_map.shape[1]:
            # print(f"Pelvis UV {(u,v)} out of depth map bounds {depth_map.shape}")
            return None
        return pelvis_uv
    else:
        if v >= depth_map.shape[0] or u >= depth_map.shape[1]:
            # print(f"Pelvis UV {(u,v)} out of depth map bounds {depth_map.shape}")
            return None

        depth = depth_map[v, u].item() * 5  # Scale depth value
        if depth <= 0:
            # print(f"Invalid depth value at UV {(u,v)}: {depth}")
            return None
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

        pelvis_3d = torch.tensor([
            (u - cx) * depth / fx,
            (v - cy) * depth / fy,
            depth
        ], dtype=torch.float32).to(img_tensor.device)

        return pelvis_3d