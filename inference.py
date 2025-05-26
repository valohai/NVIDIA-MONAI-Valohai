"""
Module for running inference with liver segmentation model.
"""
import numpy as np
import torch
from monai.transforms import (
    Compose, Invertd, AsDiscreted, SaveImaged
)
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, decollate_batch
from utils.model import get_model_network
from utils.transforms import get_transforms


def run_inference(ckpt, input_image_path, output_path):
    """
    Run inference on a single liver image.
    
    Args:
        ckpt (str): Path to the model checkpoint
        input_image_path (str): Path to input image
        output_path (str): Path to save segmentation mask
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = get_model_network()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    # Define transforms
    inference_transform = get_transforms('inference')
    
    # Attach output dir to post-transforms
    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=inference_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),
        SaveImaged(
            keys="pred",
            meta_keys="image_meta_dict",
            output_postfix="pred",
            output_dir=output_path,
            separate_folder=False,
            resample=False,
            output_dtype=np.uint8,  # Save as uint8 for segmentation masks
        )
    ])

    # Prepare input
    test_data = [{"image": input_image_path}]
    test_ds = Dataset(data=test_data, transform=inference_transform)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

    # Inference loop
    with torch.no_grad():
        for batch in test_loader:
            test_inputs = batch["image"].to(device)
            test_outputs = sliding_window_inference(test_inputs, (160, 160, 160), 4, model)

            # Prepare for inversion and saving
            decollated_outputs = decollate_batch(test_outputs)
            batch_data = []
            for i, pred in enumerate(decollated_outputs):
                batch_data.append({
                    "pred": pred,
                    "image": batch["image"][i], 
                    "pred_meta_dict": batch["image_meta_dict"],
                    "image_meta_dict": batch["image_meta_dict"]
                })

            # Apply post transforms (inversion + save)
            for data_dict in batch_data:
                post_transforms(data_dict)

    print(f"Segmentation mask saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run liver segmentation inference')
    parser.add_argument('--input_path', type=str, default='liver_201.nii.gz')
    parser.add_argument('--output_path', type=str, default='predictions/')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_metirc_model.pth')
    args = parser.parse_args()

    run_inference(args.model_path, args.input_path, args.output_path)
