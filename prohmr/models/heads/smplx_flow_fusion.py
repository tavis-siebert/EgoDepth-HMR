import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Optional, Dict, Tuple
from nflows.flows import ConditionalGlow
from yacs.config import CfgNode

from prohmr.utils.geometry import rot6d_to_rotmat
from .fc_head_smplx import FCHeadSMPLX


class SMPLXFlowFusion(nn.Module):

    def __init__(self, cfg: CfgNode, context_feats_dim=None):
        """
        Probabilistic SMPL head using Normalizing Flows.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(SMPLXFlowFusion, self).__init__()
        self.cfg = cfg
        self.npose = 6*(21 + 1)
        self.flow_surfnorms = ConditionalGlow(132, cfg.MODEL.FLOW.LAYER_HIDDEN_FEATURES,
                                    cfg.MODEL.FLOW.NUM_LAYERS, cfg.MODEL.FLOW.LAYER_DEPTH,
                                    context_features=context_feats_dim)
        self.flow_depth = ConditionalGlow(132, cfg.MODEL.FLOW.LAYER_HIDDEN_FEATURES,
                                    cfg.MODEL.FLOW.NUM_LAYERS, cfg.MODEL.FLOW.LAYER_DEPTH,
                                    context_features=context_feats_dim)
        self.fc_head = FCHeadSMPLX(cfg, context_feats_dim * 2)
        
        self.pose_fusion_fc = nn.Sequential(
            nn.Linear(132 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 132)
        )
        
        self.z_fusion_fc = nn.Sequential(
            nn.Linear(132 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 132)
        )


    # Autocasting is disabled because SMPL has numerical instability issues with fp16 parameters.
    @autocast(enabled=False)
    def log_prob(self, smpl_params: Dict, feats_surfnorms: torch.Tensor, feats_depth: torch.Tensor) -> Tuple:
        """
        Compute the log-probability of a set of smpl_params given a batch of images.
        Args:
            smpl_params (Dict): Dictionary containing a set of SMPL parameters.
            feats (torch.Tensor): Conditioning features of shape (N, C).
        Returns:
            log_prob (torch.Tensor): Log-probability of the samples with shape (B, N).
            z (torch.Tensor): The Gaussian latent corresponding to each sample with shape (B, N, 132).
        """

        feats_surfnorms = feats_surfnorms.float()
        feats_depth = feats_depth.float()
        batch_size = feats_surfnorms.shape[0]
        samples = torch.cat((smpl_params['global_orient'], smpl_params['body_pose']), dim=-1)  # [bs, 1, 132]
        num_samples = samples.shape[1]
        feats_surfnorms = feats_surfnorms.reshape(batch_size, 1, -1).repeat(1, num_samples, 1)  # [bs, 1, 2048]
        feats_depth = feats_depth.reshape(batch_size, 1, -1).repeat(1, num_samples, 1)  # [bs, 1, 2048]
        
        # Flatten for flow input
        flat_samples = samples.reshape(batch_size * num_samples, -1)
        flat_feats_surfnorms = feats_surfnorms.reshape(batch_size * num_samples, -1)
        flat_feats_depth = feats_depth.reshape(batch_size * num_samples, -1)
        
        # Compute log_prob and z for both flows
        log_prob_surf, z_surf = self.flow_surfnorms.log_prob(flat_samples, flat_feats_surfnorms)
        log_prob_depth, z_depth = self.flow_depth.log_prob(flat_samples, flat_feats_depth)

        # Reshape outputs
        log_prob_surf = log_prob_surf.reshape(batch_size, num_samples)
        log_prob_depth = log_prob_depth.reshape(batch_size, num_samples)
        z_surf = z_surf.reshape(batch_size, num_samples, -1)
        z_depth = z_depth.reshape(batch_size, num_samples, -1)

        # Combine log probs (assuming independence)
        log_prob = log_prob_surf + log_prob_depth

        # Fuse latent codes
        z_concat = torch.cat([z_surf, z_depth], dim=-1)  # [B, N, 264]
        z_fused = self.z_fusion_fc(z_concat)             # [B, N, 132]

        return log_prob, z_fused

    @autocast(enabled=False)
    def forward(self, feats_surfnorms: torch.Tensor, feats_depth: torch.Tensor, num_samples: Optional[int] = None, z: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple:
        """
        Run a forward pass of the model.
        If z is not specified, then the model randomly draws num_samples samples for each image in the batch.
        Otherwise the batch of latent vectors z is transformed using the Conditional Normalizing Flows model.
        Args:
            feats (torch.Tensor): Conditioning features of shape (N, C).
            num_samples (int): Number of samples to draw per image.
            z (torch.Tensor): A batch of latent vectors of shape (B, N, 144).
        Returns:
            pred_smpl_params (Dict): Dictionary containing the predicted set of SMPL parameters.
            pred_cam (torch.Tensor): Predicted camera parameters with shape (B, N, 3).
            log_prob (torch.Tensor): Log-probability of the samples with shape (B, N).
            z (torch.Tensor): Either the input z or the randomly drawn batch of latent Gaussian vectors.
            pred_pose_6d (torch.Tensor): Predicted pose vectors in the 6-dimensional representation.
        """

        feats_surfnorms = feats_surfnorms.float()
        feats_depth = feats_depth.float()
        
        feats = torch.cat((feats_surfnorms, feats_depth), dim=-1)  # [B, N, 2048]

        batch_size = feats_depth.shape[0]

        if z is None:
            # Sample from both flows    
            samples_surf, log_prob_surf, z_surf = self.flow_surfnorms.sample_and_log_prob(num_samples, context=feats_surfnorms)
            samples_depth, log_prob_depth, z_depth = self.flow_depth.sample_and_log_prob(num_samples, context=feats_depth)
        else:
            z_surf, z_depth = z
            num_samples = z_surf.shape[1]
            samples_surf, log_prob_surf, z_surf = self.flow_surfnorms.sample_and_log_prob(num_samples, context=feats_surfnorms, noise=z_surf)
            samples_depth, log_prob_depth, z_depth = self.flow_depth.sample_and_log_prob(num_samples, context=feats_depth, noise=z_depth)


        # Combine log_probs (assuming independence)
        log_prob = log_prob_surf + log_prob_depth

        # Extract 6D pose predictions
        pose_surf = samples_surf[:, :, :self.npose]    # [B, N, 132]
        pose_depth = samples_depth[:, :, :self.npose]  # [B, N, 132]
        
        # Concatenate and fuse to [B, N, 132]
        pose_concat = torch.cat([pose_surf, pose_depth], dim=-1)  # [B, N, 264]
        pred_pose = self.pose_fusion_fc(pose_concat)           # [B, N, 132]
        pred_pose_6d = pred_pose.clone()
        
        pred_pose = rot6d_to_rotmat(pred_pose.reshape(batch_size * num_samples, -1)).view(batch_size, num_samples, 21+1, 3, 3)  # [bs, 1/num_sample, 21, 3, 3]
        pred_smpl_params = {'global_orient': pred_pose[:, :, [0]],  # [bs, 1/num_sample, 1, 3, 3]
                             'body_pose': pred_pose[:, :, 1:]}  # [bs, 1/num_sample, 21, 3, 3]
        pred_betas, pred_cam = self.fc_head(pred_smpl_params, feats)
        pred_smpl_params['betas'] = pred_betas

        return pred_smpl_params, pred_cam, log_prob, z, pred_pose_6d
