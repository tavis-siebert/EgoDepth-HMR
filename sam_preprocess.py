import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from pelvis import estimate_pelvis_from_surface_normals
from prohmr.datasets.image_dataset_surfnormals_egobody_new import ImageDatasetSurfnormalsEgoBody
import os
import cv2
from prohmr.configs import get_config

def process_surface_normals_dataset(dataloader, sam_checkpoint, model_type="vit_h"):
    """
    Processes a dataset of surface normals and depth maps to predict person masks using SAM2 and pelvis point prompts.

    Args:
        dataloader (DataLoader): Yields dicts with keys 'surf_normals' (B, 3, H, W), 'img' (B, H, W)
        sam_checkpoint (str): Path to SAM2 checkpoint.
        model_type (str): Type of SAM2 model (e.g., "vit_h").

    Returns:
        List[np.ndarray]: List of predicted masks (H, W, bool) per sample.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SAM2 and wrap predictor
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    all_masks = []

    scale = 1
    width = 320 * scale
    height = 288 * scale
    focal_length = 200 * scale
    intrinsic_matrix = torch.tensor([
        [focal_length, 0, width // 2],
        [0, focal_length, height // 2],
        [0, 0, 1.0]
    ], dtype=torch.float32)

    for batch in tqdm(dataloader, desc="Processing Batches"):
        surf_normals_batch = batch['surf_normals']  # (B, 3, H, W)
        depth_batch = batch['img']                 # (B, H, W)
        B = surf_normals_batch.shape[0]

        for i in range(B):
            surface_normal_img = surf_normals_batch[i]
            depth_map = depth_batch[i]

            pelvis_uv = estimate_pelvis_from_surface_normals(
                surface_normal_img, depth_map, intrinsic_matrix, False
            )

            # Convert surface normal image to numpy for SAM2
            surf_np = surface_normal_img.permute(1, 2, 0).cpu().numpy()
            surf_img = ((surf_np + 1) * 127.5).astype(np.uint8)

            predictor.set_image(surf_img)

            if pelvis_uv is not None:
                point = np.array([pelvis_uv])
                label = np.array([1])
                masks, _, _ = predictor.predict(
                    point_coords=point,
                    point_labels=label,
                    multimask_output=False
                )
                all_masks.append(masks[0])
            else:
                # fallback: zero mask
                all_masks.append(np.zeros((surf_img.shape[0], surf_img.shape[1]), dtype=bool))

    return all_masks

if __name__ == "__main__":
    import argparse 

    parser = argparse.ArgumentParser(description="Process surface normals with SAM2 and pelvis prompts.")
    parser.add_argument("--sam_checkpoint", type=str, default='/work/courses/digital_human/13/weiwan/sam_vit_h_4b8939.pth', help="Path to SAM2 checkpoint.")
    parser.add_argument('--dataset_root', type=str, default='/work/courses/digital_human/13/egobody_release')
    parser.add_argument('--checkpoint', type=str, default='try_egogen_new_data/92990/best_model.pt')  # runs_try/90505/best_model.pt data/checkpoint.pt
    parser.add_argument('--model_cfg', type=str, default="prohmr/configs/prohmr_fusion.yaml", help='Path to config file. If not set use the default (prohmr/configs/prohmr_fusion.yaml)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=2, help='Number of test samples to draw')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
    parser.add_argument('--log_freq', type=int, default=100, help='How often to log results')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--shuffle', default='False', type=lambda x: x.lower() in ['true', '1'])  # todo
    parser.add_argument('--output_dir', type=str, default='/work/courses/digital_human/13/weiwan/masks', help='Directory to save output masks')
    parser.add_argument('--model_type', type=str, default='vit_h', choices=['vit_b', 'vit_l', 'vit_h'], help='Type of SAM2 model to use')
    
    
    args = parser.parse_args()
    
    model_cfg = get_config(args.model_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = ImageDatasetSurfnormalsEgoBody(cfg=model_cfg, train=False, device=device, img_dir=args.dataset_root,
                                       dataset_file=os.path.join(args.dataset_root, 'smplx_spin_holo_depth_npz/egocapture_test_smplx_split_known.npz'),
                                    #    dataset_file = "./data/smplx_spin_npz/egocapture_test_smplx_depth_top5.npz",
                                       spacing=1, split='test')
    dataloader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    masks = process_surface_normals_dataset(dataloader, args.sam_checkpoint, args.model_type)
    
    # Save or process masks as needed
    print(f"Processed {len(masks)} masks.")
    
    # Example: Save masks to output directory
    os.makedirs(args.output_dir, exist_ok=True)
    for idx, mask in enumerate(masks):
        mask_path = os.path.join(args.output_dir, f"mask_{idx}.png")
        # Save mask as an image (assuming mask is a boolean array)
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255  # Convert to binary image
        cv2.imwrite(mask_path, mask)
        print(f"Saved mask {idx} to {mask_path}")
# Note: Ensure you have the necessary imports and dataset class defined.
