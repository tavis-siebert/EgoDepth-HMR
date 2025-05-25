import torch
import os
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2

def human_segmentation(batch_idx: int, imgsz = 224, device='cuda'):
    """
    Args:
        batch_idx (int): Index of the batch to save
        device (str): 'cuda' or 'cpu'

    Returns:
        List[np.ndarray]: A list of binary masks, each shape (H, W)
    """
    
    checkpoint = "./thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "./thirdparty/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    binary_masks = []
    B = images.size(0)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(B):
            # images (torch.Tensor): Tensor of shape [B, 3, H, W] in range [0, 1] or [0, 255]
            # image_tensor = images[i]
            image_tensor = cv2.imread(f'/work/courses/digital_human/13/kotaik/depth_imgs/{batch_idx:02}/{i:02}.png')
            predictor.set_image(image_tensor)
            masks, _, _ = predictor.predict(multimask_output=False,
                                            # point_coords=input_point,
                                            # point_labels=input_label,
                                            )
            # cv2.imwrite('/home/kotaik/DH/EgoDepth-HMR/output_sam2/depth.png', image_tensor.cpu().numpy())
            cv2.imwrite('home/kotaik/DH/EgoDepth-HMR/output_sam2/mask.png', mask * 255)
        
    # Convert to tensor
    binary_masks = torch.stack(binary_masks).to(device)
    print("shape of binary masks: ",binary_masks.shape)
        
    return binary_masks

def save_depth_imgs(images: torch.Tensor, batch_idx: int, imgsz = 224, device='cuda'):
    """
    Args:
        images (torch.Tensor): Tensor of shape [B, 3, H, W] in range [0, 1] or [0, 255]
        batch_idx (int): Index of the batch to save
        device (str): 'cuda' or 'cpu'
    """
    B = images.size(0)
    os.mkdir(f'/work/courses/digital_human/13/kotaik/depth_imgs/{batch_idx:02}')
    for i in range(B):
        image_tensor = images[i]
        cv2.imwrite(f'/work/courses/digital_human/13/kotaik/depth_imgs/{batch_idx:02}/{i:02}.png', image_tensor.cpu().numpy())
