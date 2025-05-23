"""
Module for running inference with liver segmentation model.
"""
import os
import torch
import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    ScaleIntensityd, EnsureTyped, ResizeWithPadOrCropd,Orientationd,ScaleIntensityRanged,CropForegroundd,Invertd,AsDiscreted,SaveImaged
)
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import Dataset, DataLoader
from monai.data import decollate_batch
from monai.handlers.utils import from_engine
import matplotlib.pyplot as plt
from monai.networks.nets import UNet

def run_inference(ckpt, input_image_path, output_path):
    """
    Run inference on a single liver image.
    
    Args:
        model: Trained segmentation model
        input_image_path (str): Path to input image
        output_path (str): Path to save segmentation mask
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.to(device)
    model.eval()

    # Define preprocessing transforms
    test_org_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        ]
    )

    post_transforms = Compose(
        [
            # Invertd(
            #     keys="pred",
            #     transform=test_org_transforms,
            #     orig_keys="image",
            #     meta_keys="pred_meta_dict",
            #     orig_meta_keys="image_meta_dict",
            #     meta_key_postfix="meta_dict",
            #     nearest_interp=False,
            #     to_tensor=True,
            # ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
        ]
    )
    # Prepare input data
    test_data = [{"image": input_image_path}]
    
    # Apply transforms
    test_org_ds = Dataset(data=test_data, transform=test_org_transforms)

    test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)


    # Run inference
    model.load_state_dict(torch.load(os.path.join(ckpt), weights_only=True))
    model.eval()

    with torch.no_grad():
        for test_data in test_org_loader:
            test_inputs = test_data["image"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

    #         # uncomment the following lines to visualize the predicted results

            from monai.transforms import LoadImage
            loader = LoadImage()
            test_output = from_engine(["pred"])(test_data)
            

            original_image = loader(test_output[0].meta["filename_or_obj"])

            plt.figure("check", (18, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image[:, :, 20], cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(test_output[0].detach().cpu()[1, :, :, 20])
            plt.show()

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