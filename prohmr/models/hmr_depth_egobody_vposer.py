import torch
import torch.nn as nn
import smplx
import sys
import os
from typing import Any, Dict, Tuple

# Make sure to clone the human_body_prior repo and run pip install -e . in the base folder
# you might have to remove __init__.py if it exists
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

from yacs.config import CfgNode
# import open3d as o3d
from prohmr.models.backbones.resnet_depth import resnet
from prohmr.utils.geometry import aa_to_rotmat, rot6d_to_rotmat
from prohmr.utils.konia_transform import rotation_matrix_to_angle_axis
# from prohmr.optimization import OptimizationTask
from .heads import ResBlock
from .losses import Keypoint3DLoss
from ..utils.renderer import *

class HMRDepthEgoBodyVPoser(nn.Module):

    def __init__(self, cfg: CfgNode, device=None, writer=None, logger=None, with_global_3d_loss=False):
        """
        Setup HMR + VPoser model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super(HMRDepthEgoBodyVPoser, self).__init__()

        self.cfg = cfg
        self.device = device
        self.writer = writer
        self.logger = logger

        self.with_global_3d_loss = with_global_3d_loss

        # self.backbone = create_backbone(cfg).to(self.device)
        self.backbone = resnet().to(self.device)

        # VPoser
        #NOTE load_model() is a little too generic, I might want to rework this at some point
        self.pose_net, _ = load_model(
            cfg.MODEL.VPOSER.EXPR_DIR, 
            model_code=VPoser,
            remove_words_in_model_weights='vp_model.',
            disable_grad=True
        )
        self.pose_net.to(self.device)
        # manially freeze it just to make sure
        for p in self.pose_net.parameters():
            p.requires_grad = False

        # Map from conditioning feature to latent (for VPOser decoding)
        self.z_net = nn.Sequential(
            nn.LazyLinear(768),
            nn.ReLU(),
            ResBlock(768),
            ResBlock(768),
            nn.Linear(768, 32),
        ).to(self.device)

        # Beta regressor
        self.beta_net = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ).to(self.device)

        # Camera regressor
        self.cam_net = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        ).to(self.device)

        # Global orientation regressor
        self.glob_orient_net = nn.Sequential(
            nn.LazyLinear(256), 
            nn.ReLU(),
            nn.Linear(256, 6)
        ).to(self.device)

        # Create discriminator
        # self.discriminator = Discriminator().to(self.device)

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.v2v_loss = nn.L1Loss(reduction='none')
        self.smpl_parameter_loss = nn.MSELoss(reduction='none')

        # Instantiate SMPL model
        # smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        # self.smpl = SMPL(**smpl_cfg).to(self.device)
        self.smplx = smplx.create('data/smplx_model', model_type='smplx', gender='neutral', ext='npz').to(self.device)
        self.smplx_male = smplx.create('data/smplx_model', model_type='smplx', gender='male', ext='npz').to(self.device)
        self.smplx_female = smplx.create('data/smplx_model', model_type='smplx', gender='female', ext='npz').to(self.device)

    def init_optimizers(self):
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        model_params = list(self.backbone.parameters()) +\
                        list(self.z_net.parameters()) +\
                        list(self.beta_net.parameters()) +\
                        list(self.cam_net.parameters()) +\
                        list(self.glob_orient_net.parameters())

        self.optimizer = torch.optim.AdamW(
            params=model_params,
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY
        )

        '''self.optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                           lr=self.cfg.TRAIN.LR,
                                           weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)'''
        # return optimizer, optimizer_disc

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        num_samples = 1 # we are not sampling over a distribution anymore

        # Use RGB image as input
        x = batch['img'].unsqueeze(1)  # [bs, 1, 224, 224]
        batch_size = x.shape[0]

        # Compute keypoint features using the backbone
        conditioning_feats = self.backbone(x)  # [bs, 2048]

        # Predict pose
        z = self.z_net(conditioning_feats)
        pred_body_pose = self.pose_net.decoder_net(z).view(batch_size, -1, 3, 3).unsqueeze(1)

        # Predict global_orientation, betas, and camera
        pred_global_orient = self.glob_orient_net(conditioning_feats)
        pred_global_orient = rot6d_to_rotmat(pred_global_orient).unsqueeze(1)

        pred_betas = self.beta_net(conditioning_feats).unsqueeze(1)
        pred_cam = self.cam_net(conditioning_feats).unsqueeze(1)
        
        pred_smpl_params = {}
        pred_smpl_params['body_pose'] = pred_body_pose
        pred_smpl_params['betas'] = pred_betas
        pred_smpl_params['global_orient'] = pred_global_orient

        # Store useful regression outputs to the output dict
        output = {}
        output['latent_code'] = z.unsqueeze(1)
        output['pred_cam'] = pred_cam  # [bs, num_sample, 3]
        #  global_orient: [bs, num_sample, 1, 3, 3], body_pose: [bs, num_sample, 23, 3, 3], shape...
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size * num_samples, -1, 3, 3)
        pred_smpl_params['global_orient'] = rotation_matrix_to_angle_axis(pred_smpl_params['global_orient'].reshape(-1, 3, 3)).reshape(batch_size * num_samples, -1, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1, 3, 3)
        pred_smpl_params['body_pose'] = rotation_matrix_to_angle_axis(pred_smpl_params['body_pose'].reshape(-1, 3, 3)).reshape(batch_size * num_samples, -1, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size * num_samples, -1)
  
        self.smplx = smplx.create('data/smplx_model', model_type='smplx', gender='neutral', ext='npz', batch_size=pred_smpl_params['global_orient'].shape[0]).to(self.device)
        smplx_output = self.smplx(**{k: v.float() for k,v in pred_smpl_params.items()})
        pred_keypoints_3d = smplx_output.joints  # [bs*num_sample, 127, 3]
        pred_vertices = smplx_output.vertices  # [bs*num_sample, 10475, 3]
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, num_samples, -1, 3)  # [bs, num_sample, 127, 3]
        output['pred_vertices'] = pred_vertices.reshape(batch_size, num_samples, -1, 3)  # [bs, num_sample, 12475, 3]
        output['pred_keypoints_3d_global'] = output['pred_keypoints_3d'] + output['pred_cam'].unsqueeze(-2)  # [bs, n_sample, 127, 3]

        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_smpl_params = output['pred_smpl_params']
        # pred_keypoints_3d = output['pred_keypoints_3d'][:, :, 0:25]  # [bs, n_sample, 25, 3]
        pred_keypoints_3d_global = output['pred_keypoints_3d_global'][:, :, 0:22]
        pred_keypoints_3d = output['pred_keypoints_3d'][:, :, 0:22]

        batch_size = pred_smpl_params['body_pose'].shape[0]
        num_samples = pred_smpl_params['body_pose'].shape[1]

        gt_keypoints_3d_global = batch['keypoints_3d'][:, 0:22]  # [bs, 22, 3]
        gt_smpl_params = batch['smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']
        gt_gender = batch['gender']

        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d_global, gt_keypoints_3d_global.unsqueeze(1).repeat(1, num_samples, 1, 1), pelvis_id=0, pelvis_align=True)  # [bs, n_sample]
        loss_keypoints_3d_full = self.keypoint_3d_loss(pred_keypoints_3d_global, gt_keypoints_3d_global.unsqueeze(1).repeat(1, num_samples, 1, 1), pelvis_align=False)

        # loss_transl = F.l1_loss(output['pred_cam_t_full'], gt_smpl_params['transl'].unsqueeze(1).repeat(1, num_samples, 1), reduction='mean')

        ####### compute v2v loss
        temp_bs = gt_smpl_params['body_pose'].shape[0]
        self.smplx_male = smplx.create('data/smplx_model', model_type='smplx', gender='male', ext='npz', batch_size=temp_bs).to(self.device)
        self.smplx_female = smplx.create('data/smplx_model', model_type='smplx', gender='female', ext='npz', batch_size=temp_bs).to(self.device)
        gt_smpl_output = self.smplx_male(**{k: v.float() for k, v in gt_smpl_params.items()})
        gt_vertices = gt_smpl_output.vertices  # smplx vertices
        gt_joints = gt_smpl_output.joints
        gt_smpl_output_female = self.smplx_female(**{k: v.float() for k, v in gt_smpl_params.items()})
        gt_vertices_female = gt_smpl_output_female.vertices
        gt_joints_female = gt_smpl_output_female.joints
        gt_vertices[gt_gender == 1, :, :] = gt_vertices_female[gt_gender == 1, :, :]  # [bs, 10475, 3]
        gt_joints[gt_gender == 1, :, :] = gt_joints_female[gt_gender == 1, :, :]

        gt_vertices = gt_vertices.unsqueeze(1).repeat(1, num_samples, 1, 1)  # [bs, n_sample, 10475, 3]
        gt_pelvis = gt_joints[:, [0], :].clone().unsqueeze(1).repeat(1, num_samples, 1, 1)  # [bs, n_sample, 1, 3]
        pred_vertices = output['pred_vertices']  # [bs, num_sample, 10475, 3]
        loss_v2v = self.v2v_loss(pred_vertices - pred_keypoints_3d[:, :, [0], :].clone(), gt_vertices - gt_pelvis).mean(dim=(2, 3))  # [bs, n_sample]
        loss_v2v = loss_v2v[:, [0]].mean()  # avg over batch, vertices

        # Compute loss on SMPL parameters
        # loss_smpl_params: keys: ['global_orient', 'body_pose', 'betas']
        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            if k != 'transl':
                gt = gt_smpl_params[k].unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
                if is_axis_angle[k].all():
                    gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size * num_samples, -1, 3, 3)
                # has_gt = has_smpl_params[k].unsqueeze(1).repeat(1, num_samples)
                loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, num_samples, -1), gt.reshape(batch_size, num_samples, -1))


        loss_keypoints_3d = loss_keypoints_3d[:, [0]].sum() / batch_size
        loss_keypoints_3d_full = loss_keypoints_3d_full[:, [0]].sum() / batch_size

        loss_smpl_params = {k: v[:, [0]].sum() / batch_size for k,v in loss_smpl_params.items()}

        # Filter out images with corresponding SMPL parameter annotations
        # smpl_params = {k: v.clone() for k,v in gt_smpl_params.items()}
        smpl_params = {k: v.clone() for k, v in gt_smpl_params.items() if k!='transl'}
        smpl_params['body_pose'] = aa_to_rotmat(smpl_params['body_pose'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)  # [bs, 1,126]
        smpl_params['global_orient'] = aa_to_rotmat(smpl_params['global_orient'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)
        smpl_params['betas'] = smpl_params['betas'].unsqueeze(1)
        # has_smpl_params = (batch['has_smpl_params']['body_pose'] > 0)
        smpl_params = {k: v for k, v in smpl_params.items()}

        # Add some noise to annotations at training time to prevent overfitting
        if train:
            smpl_params = {k: v + self.cfg.TRAIN.SMPL_PARAM_NOISE_RATIO * torch.randn_like(v) for k, v in smpl_params.items()}

        # Latent loss
        z = output['latent_code']
        loss_latent = (z.squeeze(1)**2).sum(dim=1).mean()

        loss = self.cfg.LOSS_WEIGHTS['LATENT'] * loss_latent +\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_MODE'] * loss_keypoints_3d +\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_FULL_MODE'] * loss_keypoints_3d_full * self.with_global_3d_loss +\
               self.cfg.LOSS_WEIGHTS['V2V_MODE'] * loss_v2v +\
               sum([loss_smpl_params[k] * self.cfg.LOSS_WEIGHTS[(k+'_MODE').upper()] for k in loss_smpl_params])

        losses = dict(
            loss=loss.detach(),
            loss_keypoints_3d_mode=loss_keypoints_3d.detach(),
            loss_keypoints_3d_full_mode=loss_keypoints_3d_full.detach(),
            loss_v2v_mode=loss_v2v.detach(),
            loss_latent=loss_latent.detach()
        )
        
        output['losses'] = losses

        return loss


    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch, train=False)


    def training_step(self, batch: Dict, mocap_batch: Dict) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        ### read input data
        # batch = joint_batch['img']   # [64, 3, 224, 224]
        # mocap_batch = joint_batch['mocap']
        # optimizer, optimizer_disc = self.optimizers(use_pl_optimizer=True)

        self.backbone.train()
        self.z_net.train()
        self.glob_orient_net.train()
        self.cam_net.train()
        self.beta_net.train()

        ### G forward step
        output = self.forward_step(batch, train=True)
        ### compute G loss
        loss = self.compute_loss(batch, output, train=True)
        self.optimizer.zero_grad()
        # self.manual_backward(loss)
        loss.backward()
        self.optimizer.step()

        return output

    def validation_step(self, batch: Dict) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """

        self.backbone.eval()
        self.z_net.eval()
        self.glob_orient_net.eval()
        self.cam_net.eval()
        self.beta_net.eval()

        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        return output
