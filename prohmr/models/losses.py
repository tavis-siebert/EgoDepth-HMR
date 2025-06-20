import torch
import torch.nn as nn

import torch
import torch.nn as nn

from ..utils.geometry import render_keypoints_to_depth_map_fast

class DepthMapLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1'):
        """
        Depth map loss module using sparse depth from reprojected keypoints.
        Args:
            loss_type (str): Choose between 'l1' and 'l2' losses.
        """
        super(DepthMapLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError(f"Unsupported loss_type: {loss_type}")

    def forward(self,
                keypoints_2d: torch.Tensor,         # [B, N, 2]
                keypoints_depth: torch.Tensor,      # [B, N]
                gt_depth_map: torch.Tensor,         # [B, H, W]
                gt_depth_mask: torch.Tensor,         # [B, H, W]
                ) -> torch.Tensor:
        """
        Compute sparse depth loss from keypoints against ground truth depth map.
        
        Returns:
            loss: (B,) per-batch scalar depth loss
        """
        # print("depth shape: ", keypoints_depth.shape)
        B, H, W = gt_depth_map.shape
        device = keypoints_depth.device

        # Rasterize predicted sparse depth map and valid mask
        pred_depth_map, pred_mask = render_keypoints_to_depth_map_fast(
            keypoints_2d, keypoints_depth, (H, W)
        )
        
        pred_depth_map = pred_depth_map.view(B, -1, H, W)  # [B, N, H, W]
        pred_mask = pred_mask.view(B, -1, H, W)  # [B, N, H, W]

        # Mask out invalid pixels from gt
        gt_valid_mask = (gt_depth_map > 0) & (gt_depth_map < 5)
        gt_valid_mask = gt_valid_mask.view(B, -1, H, W)  # [B, 1, H, W]
        N = pred_depth_map.shape[1]
        gt_depth_map = gt_depth_map.view(B, -1, H, W)  # [B, N, H, W]
        gt_depth_map = gt_depth_map.repeat(1, N, 1, 1)
    
        
        # print("pred_depth_map shape: ", pred_depth_map.shape)
        # print("pred_mask shape: ", pred_mask.shape)
        # print("gt_valid_mask shape: ", gt_valid_mask.shape)

        # Combined valid mask
        valid_mask = pred_mask & gt_valid_mask  # [B, N, H, W]
        valid_mask = gt_depth_mask

        # Compute per-pixel depth loss
        loss_map = self.loss_fn(pred_depth_map, gt_depth_map)  # [B, N, H, W]
        # print("loss_map shape: ", loss_map.shape)

        # Zero out invalid regions
        loss_map[~valid_mask] = 0.0

        # Aggregate loss per batch
        loss = loss_map.sum(dim=(-2, -1)) / (valid_mask.sum(dim=(-2, -1)) + 1e-8)  # [B, N]
        # print("loss shape: ", loss.shape)
        
        valid_mask_sum = valid_mask.sum(dim=(-2, -1))  # [B, N]
        valid_ratio = valid_mask_sum / (H * W)  # [B, N]
        valid_penalty = torch.clamp(1 - valid_ratio, min=0.0)  # [B, N]
        valid_regularization = valid_penalty * 0.1  # [B, N]
        
        loss += valid_regularization  # [B, N]
        
        
        
        

        return loss # shape [B]



class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor, joints_to_ign=None) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, :, -1].unsqueeze(-1).clone()  # [B, S, N, 1]
        conf[:, :, joints_to_ign, :] = 0  # todo: check
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :, :-1])).sum(dim=(2,3))
        return loss


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d=None, gt_keypoints_3d=None, pelvis_id=0, pelvis_align=False):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        if pelvis_align:
            pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, :, pelvis_id, :].unsqueeze(dim=2).clone()
            gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:, :, pelvis_id, :].unsqueeze(dim=2)
        loss = self.loss_fn(pred_keypoints_3d, gt_keypoints_3d).sum(dim=(2,3))
        return loss

class ParameterLoss(nn.Module):

    def __init__(self):
        """
        SMPL parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor):
        """
        Compute SMPL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth SMPL parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        batch_size = pred_param.shape[0]
        num_samples = pred_param.shape[1]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size, num_samples] + [1] * (num_dims-2)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = (has_param * self.loss_fn(pred_param, gt_param))
        return loss_param



class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def compute_geodesic_distance(self, m1, m2):
        """ Compute the geodesic distance between two rotation matrices.
        Args:
            m1, m2: Two rotation matrices with the shape (batch x 3 x 3).
        Returns:
            The minimal angular difference between two rotation matrices in radian form [0, pi].
        """
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
        cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

        theta = torch.acos(cos)

        return theta

    def __call__(self, m1, m2, reduction='mean'):
        loss = self.compute_geodesic_distance(m1, m2)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            raise RuntimeError(f'unsupported reduction: {reduction}')



def cal_bps_body2scene(scene_pcd, body_pcd):
    # scene_pcd: [bs, n_scene_pts, 3], body_pcd: [bs, 24/45/66/, 3]
    n_scene_pts = scene_pcd.shape[1]
    n_body_pts = body_pcd.shape[1]
    scene_pcd_repeat = scene_pcd.unsqueeze(2).repeat(1, 1, n_body_pts, 1)  # [bs, n_scene_pts, n_body_pts， 3]
    body_pcd_repeat = body_pcd.unsqueeze(1).repeat(1, n_scene_pts, 1, 1)  # [bs, n_scene_pts, n_body_pts， 3]
    dist = torch.sum(((scene_pcd_repeat - body_pcd_repeat) ** 2), dim=-1).sqrt()  # [bs, n_scene_pts, n_body_pts]
    min_dist = torch.min(dist, dim=1).values  # [bs, n_body_pts]
    return min_dist
