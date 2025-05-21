"""
Module for running inference with liver segmentation model.
"""
import os
import torch
import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    ScaleIntensityd, EnsureTyped, ResizeWithPadOrCropd
)
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

def run_inference(model, input_image_path, output_path):
    """
    Run inference on a single liver image.
    
    Args:
        model: Trained segmentation model
        input_image_path (str): Path to input image
        output_path (str): Path to save segmentation mask
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Define preprocessing transforms
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 2.0),
            mode="bilinear"
        ),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=(160, 160, 160)),
        EnsureTyped(keys=["image"]),
    ])

    # Prepare input data
    data = [{"image": input_image_path}]
    
    # Apply transforms
    batch_data = transforms(data)
    inputs = batch_data[0]["image"].unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        # Sliding window inference for large volumes
        roi_size = (160, 160, 160)
        outputs = sliding_window_inference(
            inputs, roi_size, 4, model, overlap=0.5
        )
        
        # Post-process outputs
        post_trans = AsDiscrete(threshold=0.5)
        mask = post_trans(outputs[0, 0].cpu())

    # Save segmentation mask
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask_nii = nib.Nifti1Image(mask.numpy().astype(np.uint8), np.eye(4))
    nib.save(mask_nii, output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run liver segmentation inference')
    parser.add_argument('--input_path', type=str, default='liver_dataset/images_Ts/test_image.nii.gz',
                        help='Path to input image')
    parser.add_argument('--output_path', type=str, default='predictions/output_mask.nii.gz',
                        help='Path to save segmentation mask')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                        help='Path to saved model checkpoint')
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    model = checkpoint['model_state_dict']
    
    # Run inference
    run_inference(model, args.input_path, args.output_path)
    print(f"Segmentation mask saved to: {args.output_path}")