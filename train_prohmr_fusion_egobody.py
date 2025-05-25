"""
Script used to train ProHMR.
Example usage:
python train_prohmr.py --root_dir=/path/to/experiment/folder

Running the above will use the default config file to train ProHMR as in the paper.
The code uses PyTorch Lightning for training.
"""
import os
import argparse
import torch
from tqdm import tqdm
import sys
from torch.utils.data.dataloader import default_collate
import shutil
import random
from yacs.config import CfgNode as CN
import yaml
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

from prohmr.configs import get_config, prohmr_config, dataset_config
from prohmr.models import ProHMRSurfnormalsEgobody, ProHMRFusionEgobody, ProHMRFusionFlowEgobody
from prohmr.datasets.image_dataset_surfnormals_egobody import ImageDatasetSurfnormalsEgoBody
from prohmr.datasets.mocap_dataset import MoCapDataset

from utils import *
# python train_prohmr_depth_egobody.py --data_source synthetic --train_dataset_root /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/egobody_depth_new_new/ --val_dataset_root /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/egobody_release
# python train_prohmr_depth_egobody.py --load_pretrained True --checkpoint ./data/checkpoint/depth/best_model.pt --data_source real --train_dataset_root /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/egobody_release/ --val_dataset_root /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/egobody_release

parser = argparse.ArgumentParser(description='Training code for depth input')
parser.add_argument('--gpu_id', type=int, default='0')
parser.add_argument('--load_pretrained', default='False', type=lambda x: x.lower() in ['true', '1'])  # if load pretrained model
parser.add_argument('--load_only_backbone', default='False', type=lambda x: x.lower() in ['true', '1'])  # if True, only load resnet backbone from pretrained model
parser.add_argument('--load_rgb_pretrained', default='False', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--load_depth_pretrained', default='False', type=lambda x: x.lower() in ['true', '1'])  # if load pretrained model
parser.add_argument('--load_flow_pretrained', default='False', type=lambda x: x.lower() in ['true', '1'])  # if load pretrained model
parser.add_argument('--checkpoint', type=str, default='try_egogen_new_data/76509/best_global_model.pt', help='path to saved ProHMRFusion model ckpt')  # data/checkpoint.pt
parser.add_argument('--depth_checkpoint', type=str, default='try_egogen_new_data/76509/best_global_model.pt', help='path to saved ProHMRDepth model ckpt')  # data/checkpoint.ptparser.add_argument('--model_cfg', type=str, default='prohmr/configs/prohmr.yaml', help='Path to config file')  # prohmr prohmr_onlytransl
parser.add_argument('--rgb_checkpoint', type=str, default='try_egogen_new_data/76509/best_global_model.pt', help='path to saved ProHMR model (any that works on 3 channel images)')
parser.add_argument('--flow_checkpoint', type=str, default=None, help='path to save train logs and models')  # data/checkpoint.pt
parser.add_argument('--model_cfg', type=str, default='prohmr/configs/prohmr_fusion.yaml', help='Path to config file')  # prohmr prohmr_onlytransl
parser.add_argument('--save_dir', type=str, default='tmp', help='path to save train logs and models')

parser.add_argument('--data_source', type=str, default='real')  # real/synthetic/mix
parser.add_argument('--train_dataset_root', type=str, default=None)  
parser.add_argument('--val_dataset_root', type=str, default=None)
parser.add_argument('--train_dataset_file', type=str, default=None)  
parser.add_argument('--val_dataset_file', type=str, default=None)
parser.add_argument('--mix_dataset_root', type=str) 
parser.add_argument('--mix_dataset_file', type=str)  


parser.add_argument('--batch_size', type=int, default=64)  # 64
parser.add_argument('--num_workers', type=int, default=8, help='# of dataloader num_workers')
parser.add_argument('--num_epoch', type=int, default=100, help='# of training epochs ')
parser.add_argument("--log_step", default=183, type=int, help='log after n iters')  # 500
parser.add_argument("--save_step", default=183, type=int, help='save models after n iters')  # 500

parser.add_argument('--with_global_3d_loss', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--do_augment', default='True', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--shuffle', default='True', type=lambda x: x.lower() in ['true', '1'])


args = parser.parse_args()

if args.train_dataset_file is None:
    args.train_dataset_file = '/work/courses/digital_human/13/egobody_release/smplx_spin_holo_depth_npz/egocapture_train_smplx.npz'
if args.val_dataset_file is None:
    args.val_dataset_file = '/work/courses/digital_human/13/egobody_release/smplx_spin_holo_depth_npz/egocapture_val_smplx.npz'
if args.train_dataset_root is None:
    args.train_dataset_root = "/work/courses/digital_human/13/egobody_release"
if args.val_dataset_root is None:
    args.val_dataset_root = "/work/courses/digital_human/13/egobody_release"
if args.flow_checkpoint is None:
    args.flow_checkpoint = args.depth_checkpoint
torch.cuda.set_device(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('gpu id:', torch.cuda.current_device())



def collate_fn(item):
    try:
        item = default_collate(item)
    except Exception as e:
        import pdb;
        pdb.set_trace()
    return item



def train(writer, logger):
    model_cfg = get_config(args.model_cfg)

    if args.data_source != 'mix':
        train_dataset = ImageDatasetSurfnormalsEgoBody(cfg=model_cfg, train=True, device=device, img_dir=args.train_dataset_root,
                                            dataset_file=args.train_dataset_file,
                                            do_augment=args.do_augment,
                                            split='train', data_source=args.data_source)
    # else:
    #     train_dataset = ImageDatasetDepthMix(cfg=model_cfg, train=True, device=device, 
    #                                          real_img_dir=args.train_dataset_root,
    #                                          syn_img_dir=args.mix_dataset_root,
    #                                         real_dataset_file=args.train_dataset_file,
    #                                         syn_dataset_file=args.mix_dataset_file,
    #                                         do_augment=args.do_augment,
    #                                         split='train', data_source=args.data_source)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, collate_fn=collate_fn)
    train_dataloader_iter = iter(train_dataloader)


    val_dataset = ImageDatasetSurfnormalsEgoBody(cfg=model_cfg, train=False, device=device, img_dir=args.val_dataset_root,
                                           dataset_file=args.val_dataset_file,
                                           spacing=1, split='val', data_source='real')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    mocap_dataset = MoCapDataset(dataset_file='/work/courses/digital_human/13/data/datasets/cmu_mocap.npz')
    mocap_dataloader = torch.utils.data.DataLoader(mocap_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    mocap_dataloader_iter = iter(mocap_dataloader)


    # Setup model
    print("Fusion model: {}".format(model_cfg.MODEL.FUSION))
    if model_cfg.MODEL.FUSION == 'flow':
        model = ProHMRFusionFlowEgobody(cfg=model_cfg, device=device, writer=None, logger=None, with_global_3d_loss=args.with_global_3d_loss)
    else:
        model = ProHMRFusionEgobody(cfg=model_cfg, device=device, writer=None, logger=None, with_global_3d_loss=args.with_global_3d_loss)
    if not model_cfg.MODEL.BACKBONE.FREEZE_DEPTH:
        print('[INFO] train depth backbone')
    if not model_cfg.MODEL.BACKBONE.FREEZE_SURFNORMS:
        print('[INFO] train surfnormals backbone')
    model.train()

    # Load a previos ProHMRFusion checkpoint
    if args.load_pretrained:
        weights = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        if args.load_only_backbone:
            weights_backbone = {}
            weights_backbone['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] == 'backbone_rgb'}
            model.load_state_dict(weights_backbone['state_dict'], strict=False)
        else:
            # loaded_cfg_yaml = weights['config']
            # loaded_cfg = CN(new_allowed=True)
            # loaded_cfg.merge_from_other_cfg(CN(yaml.safe_load(loaded_cfg_yaml)))

            # if model_cfg.MODEL.FUSION != loaded_cfg.MODEL.FUSION:
            #     raise ValueError(f"Model requested fusion type {model_cfg.MODEL.FUSION} but loaded {loaded_cfg.MODEL.FUSION}")
            # elif model_cfg.MODEL.FLOW.CONTEXT_FEATURES != loaded_cfg.MODEL.FLOW.CONTEXT_FEATURES:
            #     raise ValueError(f"Model requested {model_cfg.MODEL.FLOW.CONTEXT_FEATURES} nflow feature dim but loaded {loaded_cfg.MODEL.FLOW.CONTEXT_FEATURES}")

            weights_copy = {}
            weights_copy['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] != 'smplx' and k.split('.')[0] != 'smplx_male' and k.split('.')[0] != 'smplx_female'}
            model.load_state_dict(weights_copy['state_dict'], strict=False)
        print('[INFO] pretrained model loaded from {}.'.format(args.checkpoint))
        print('[INFO] load_only_backbone: {}'.format(args.load_only_backbone))
    
    # Load the backbone from a ProhHMRDepth model
    if args.load_depth_pretrained:
        weights = torch.load(args.depth_checkpoint, map_location=lambda storage, loc: storage)
        weights_backbone = {}
        weights_backbone['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] == 'backbone'}
        # change the name of the key to match the current model
        weights_backbone['state_dict'] = {k.replace('backbone.', 'backbone_depth.'): v for k, v in weights_backbone['state_dict'].items()}
        model.backbone_depth.load_state_dict(weights_backbone['state_dict'], strict=False)
    if args.load_flow_pretrained:
        weights = torch.load(args.flow_checkpoint, map_location=lambda storage, loc: storage)
        weights_backbone = {}
        weights_backbone['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] == 'flow'}
        # change the name of the key to match the current model
        model.flow.load_state_dict(weights_backbone['state_dict'], strict=False)

    # Load the backbone from a ProHMRR-rgb model ('rgb' can be any 3 channel image e.g. surface normal)
    if args.load_rgb_pretrained:
        weights = torch.load(args.rgb_checkpoint, map_location=lambda storage, loc: storage)
        weights_backbone = {}
        weights_backbone['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] == 'backbone'}
        # change the name of the key to match the current model
        weights_backbone['state_dict'] = {k.replace('backbone.', 'backbone_surfnorms.'): v for k, v in weights_backbone['state_dict'].items()}
        model.backbone_surfnorms.load_state_dict(weights_backbone['state_dict'], strict=False)


    # optimizer
    model.init_optimizers()

    ################################## start training #########################################
    total_steps = 0
    best_loss_keypoints_3d_mode = 10000
    best_loss_keypoints_3d_mode_global = 10000
    for epoch in range(args.num_epoch):
        # for step, batch in tqdm(enumerate(train_dataloader)):
        #     total_steps += 1
        for step in tqdm(range(train_dataset.dataset_len // args.batch_size), desc=f"Epoch: {epoch+1}"):
            total_steps += 1

            ### iter over train loader and mocap data loader
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(train_dataloader)
                batch = next(train_dataloader_iter)

            try:
                mocap_batch = next(mocap_dataloader_iter)
            except StopIteration:
                mocap_dataloader_iter = iter(mocap_dataloader)
                mocap_batch = next(mocap_dataloader_iter)

            # import pdb; pdb.set_trace()
            for param_name in batch.keys():
                if param_name not in ['imgname', 'smpl_params', 'has_smpl_params', 'smpl_params_is_axis_angle']:
                    batch[param_name] = batch[param_name].to(device)
            for param_name in batch['smpl_params'].keys():
                batch['smpl_params'][param_name] = batch['smpl_params'][param_name].to(device)

            for param_name in mocap_batch.keys():
                mocap_batch[param_name] = mocap_batch[param_name].to(device)

            if not model_cfg.MODEL.BACKBONE.FREEZE_DEPTH and step % 2 == 0:
                output = model.training_step(batch, mocap_batch, True)
            else:
                output = model.training_step(batch, mocap_batch, False)

            ####################### log train loss ############################
            loss_curves_dir = os.path.join(writer.file_writer.get_logdir(), "loss_curves")
            if total_steps % args.log_step == 0:
                for key in output['losses'].keys():
                    writer.add_scalar('train/{}'.format(key), output['losses'][key].item(), total_steps)
                    print_str = '[Step {:d}/ Epoch {:d}] [train]  {}: {:.10f}'. \
                        format(step, epoch, key, output['losses'][key].item())
                    logger.info(print_str)
                    print(print_str)
                model.update_and_plot_losses(output['losses'], phase="train", save_dir=loss_curves_dir, plot=True)

            ####################### log val loss #################################
            if total_steps % args.log_step == 0:
                val_loss_dict = {}
                with torch.no_grad():
                    for test_step, test_batch in tqdm(enumerate(val_dataloader)):
                        last_batch = test_step == len(val_dataloader) - 1
                        for param_name in test_batch.keys():
                            if param_name not in ['imgname', 'smpl_params', 'has_smpl_params', 'smpl_params_is_axis_angle']:
                                test_batch[param_name] = test_batch[param_name].to(device)
                        for param_name in test_batch['smpl_params'].keys():
                            test_batch['smpl_params'][param_name] = test_batch['smpl_params'][param_name].to(device)

                        val_output = model.validation_step(test_batch)

                        for key in val_output['losses'].keys():
                            if test_step == 0:
                                val_loss_dict[key] = val_output['losses'][key].detach().clone()
                            else:
                                val_loss_dict[key] += val_output['losses'][key].detach().clone()

                for key in val_loss_dict.keys():
                    val_loss_dict[key] = val_loss_dict[key] / test_step
                    writer.add_scalar('val/{}'.format(key), val_loss_dict[key].item(), total_steps)
                    print_str = '[Step {:d}/ Epoch {:d}] [test]  {}: {:.10f}'. \
                        format(step, epoch, key, val_loss_dict[key].item())
                    logger.info(print_str)
                    print(print_str)
                
                model.update_and_plot_losses(val_loss_dict, phase="val", save_dir=loss_curves_dir, plot=True)

                # save model with best loss_keypoints_3d_mode
                if val_loss_dict['loss_keypoints_3d_mode'] < best_loss_keypoints_3d_mode:
                    best_loss_keypoints_3d_mode = val_loss_dict['loss_keypoints_3d_mode']
                    save_path = os.path.join(writer.file_writer.get_logdir(), "best_model.pt")
                    state = {
                        "config": model.cfg.dump(),
                        "state_dict": model.state_dict(),
                        "epoch": epoch
                    }
                    torch.save(state, save_path)
                    logger.info('[*] best model saved\n')
                    print('[*] best model saved\n')
                if val_loss_dict['loss_keypoints_3d_full_mode'] < best_loss_keypoints_3d_mode_global:
                    best_loss_keypoints_3d_mode_global = val_loss_dict['loss_keypoints_3d_full_mode']
                    save_path = os.path.join(writer.file_writer.get_logdir(), "best_global_model.pt")
                    state = {
                        "config": model.cfg.dump(),
                        "state_dict": model.state_dict(),
                        "epoch": epoch
                    }
                    torch.save(state, save_path)
                    logger.info('[*] best global model saved\n')
                    print('[*] best global model saved\n')

            ################### save trained model #######################
            if total_steps % args.save_step == 0:
                save_path = os.path.join(writer.file_writer.get_logdir(), "last_model.pt")
                state = {
                    "state_dict": model.state_dict(),
                }
                state = {
                    "config": model.cfg.dump(),
                    "state_dict": model.state_dict(),
                    "epoch": epoch
                }
                torch.save(state, save_path)
                logger.info('[*] last model saved\n')
                print('[*] last model saved\n')






if __name__ == '__main__':
    ########## set up writter, logger
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))  # create new path
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))
    sys.stdout.flush()
    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file
    save_config(logdir, args)
    # train_config_file_name = args.model_cfg.split('/')[-1]
    shutil.copyfile(args.model_cfg, os.path.join(logdir, args.model_cfg.split('/')[-1]))

    train(writer, logger)





