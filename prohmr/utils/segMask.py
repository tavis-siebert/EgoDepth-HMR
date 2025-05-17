import torch
from fastsam import FastSAM, FastSAMPrompt
from torchvision.transforms.functional import to_pil_image
import cv2

def batch_text_prompt_segmentation(images: torch.Tensor, text: str, model_path: str, imgsz = 224, device='cuda'):
    """
    Args:
        images (torch.Tensor): Tensor of shape [B, 3, H, W] in range [0, 1] or [0, 255]
        text (str): Text prompt to use (e.g., "person")
        model_path (str): Path to FastSAM checkpoint
        device (str): 'cuda' or 'cpu'

    Returns:
        List[np.ndarray]: A list of binary masks, each shape (H, W)
    """
    model = FastSAM(model_path)
    model.to(device)
    
    binary_masks = []
    B = images.size(0)
    count = 0

    for i in range(B):
        image_tensor = images[i]
        # print("shape of surfnormals: ",image_tensor.shape)
        pil_image = to_pil_image(image_tensor.permute(2, 0, 1))  # Convert to PIL.Image
        # save the image in /home/weiwan/DigitalHuman/EgoDepth-HMR/output
        cv2.imwrite(f'/home/weiwan/DigitalHuman/EgoDepth-HMR/output/surfnorm_img.png', image_tensor.cpu().numpy())
        
        # Run FastSAM (returns List[Results] with masks etc.)
        everything_results = model(
            pil_image,
            device=device,
            retina_masks=True,
            imgsz=imgsz,
            conf=0.4,
            iou=0.9
        )
        # Prompt processor
        prompt = FastSAMPrompt(pil_image, everything_results, device=device)
        mask = prompt.text_prompt(text)  # shape: (1, H, W)
        try:
            mask = mask.transpose(1, 2, 0)
            cv2.imwrite('/home/weiwan/DigitalHuman/EgoDepth-HMR/output/mask.png', mask * 255)
            binary_masks.append(mask[0])  
        except AttributeError:
            print("cannot segment mask, using the previous one")
            # save the image
            cv2.imwrite(f'/home/weiwan/DigitalHuman/EgoDepth-HMR/output/failed_img_{count}.png', image_tensor.cpu().numpy())
            binary_masks.append(binary_masks[-1])
            count += 1
        
           # Add (H, W) mask
        
    # Convert to tensor
    binary_masks = torch.stack(binary_masks).to(device)
    print("shape of binary masks: ",binary_masks.shape)
        
    return binary_masks