import os
import numpy as np
import pickle
import torch
from yacs.config import CfgNode

from .dataset import Dataset
from .utils import get_example

class BatchedImageDataset(Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 img_dir: str,
                 train: bool = False,
                 **kwargs):
        """
        Batched version of ImageDataset, where instead of a single example a list of examples is loaded (e.g. multiple views).
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """

        super(BatchedImageDataset, self).__init__()

        self.data = pickle.load(open(dataset_file, 'rb'))
        self.train = train
        self.cfg = cfg
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.img_dir = img_dir
        body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
        extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
        flip_keypoint_permutation = body_permutation + [25 + i for i in extra_permutation]
        self.flip_keypoint_permutation = flip_keypoint_permutation

    def __len__(self):
        return len(self.data)

    def total_length(self):
        """
        Return the total number of images in the dataset.
        """
        return sum([len(datum['imgname']) for datum in self.data])

    def __getitem__(self, idx: int):
        data = self.data[idx]
        num_images = len(data['imgname'])
        augm_config = self.cfg.DATASETS.CONFIG
        img_patch = []
        keypoints_2d = []
        keypoints_3d = []
        smpl_params = []
        has_smpl_params = []
        smpl_params_is_axis_angle = []
        img_size = []
        center = []
        scale = []
        if 'body_keypoints_3d' in data:
            body_keypoints_3d = data['body_keypoints_3d']
            extra_keypoints_3d = data['extra_keypoints_3d']
            keypoints_3d_all = np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1)
        else:
            keypoints_3d_all = np.zeros((num_images, 44, 4))
        for n in range(num_images):
            imgname = data['imgname'][n]
            image_file = os.path.join(self.img_dir, imgname)
            keypoints_2d_n = np.zeros((44, 3))
            keypoints_3d_n = keypoints_3d_all[n]
            center_n = data['center'][n].copy()
            center_x = center_n[0]
            center_y = center_n[1]
            bbox_size_n = 1.2*data['scale'][n]
            if 'body_pose' in data:
                body_pose_n = data['body_pose'][n]
            else:
                body_pose_n = np.zeros(72, dtype=np.float32)
            if 'betas' in data:
                betas_n = data['betas'][n]
            else:
                betas_n = np.zeros(10, dtype=np.float32)
            if 'has_body_pose' in data:
                has_body_pose_n = data['has_body_pose'][n]
            else:
                has_body_pose_n = 0.0
            if 'has_betas' in data:
                has_betas_n = data['has_betas'][n]
            else:
                has_betas_n = 0.0


            smpl_params_n = {'global_orient': body_pose_n[:3],
                            'body_pose': body_pose_n[3:],
                            'betas': betas_n
                           }

            has_smpl_params_n = {'global_orient': has_body_pose_n,
                                'body_pose': has_body_pose_n,
                                'betas': has_betas_n
                               }
            smpl_params_is_axis_angle_n = {'global_orient': True,
                                          'body_pose': True,
                                          'betas': False
                                         }
            img_patch_n, keypoints_2d_n, keypoints_3d_n, smpl_params_n, has_smpl_params_n, img_size_n = get_example(image_file,
                                                                                                          center_x, center_y,
                                                                                                          bbox_size_n, bbox_size_n,
                                                                                                          keypoints_2d_n, keypoints_3d_n,
                                                                                                          smpl_params_n, has_smpl_params_n,
                                                                                                          self.flip_keypoint_permutation,
                                                                                                          self.img_size, self.img_size,
                                                                                                          self.mean, self.std, self.train, augm_config)

            img_patch.append(img_patch_n)
            keypoints_2d.append(keypoints_2d_n)
            keypoints_3d.append(keypoints_3d_n)
            smpl_params.append(smpl_params_n)
            has_smpl_params.append(has_smpl_params_n)
            smpl_params_is_axis_angle.append(smpl_params_is_axis_angle_n)
            img_size.append(img_size_n)
        img_patch = np.stack(img_patch, axis=0)
        keypoints_2d = np.stack(keypoints_2d, axis=0)
        keypoints_3d = np.stack(keypoints_3d, axis=0)
        smpl_params = {k: np.stack([sp[k] for sp in smpl_params], axis=0) for k in smpl_params[0].keys()}
        has_smpl_params = {k: np.stack([sp[k] for sp in has_smpl_params], axis=0) for k in has_smpl_params[0].keys()}
        smpl_params_is_axis_angle = {k: np.stack([sp[k] for sp in smpl_params_is_axis_angle], axis=0) for k in smpl_params_is_axis_angle[0].keys()}
        img_size = np.stack(img_size, axis=0)

        item = {}
        item['img'] = torch.from_numpy(img_patch)
        item['keypoints_2d'] = torch.from_numpy(keypoints_2d.astype(np.float32))
        item['keypoints_3d'] = torch.from_numpy(keypoints_3d.astype(np.float32))
        item['smpl_params'] = {k: torch.from_numpy(v).float() for k,v in smpl_params.items()}
        item['has_smpl_params'] = {k: torch.from_numpy(v).bool() for k,v in has_smpl_params.items()}
        item['smpl_params_is_axis_angle'] = {k: torch.from_numpy(v).bool() for k,v in smpl_params_is_axis_angle.items()}
        return item
