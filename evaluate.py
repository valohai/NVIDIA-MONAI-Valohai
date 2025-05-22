"""
Module for evaluating liver segmentation model.
"""
import os
import torch
import argparse
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Invertd,
    AsDiscreted,
    Compose,
)
from monai.data import Dataset, DataLoader
from monai.handlers.utils import from_engine
import monai
import matplotlib.pyplot as plt

def evaluate_model(ckpt, data_dir, labels_dir, device, batch_size=1):
    """
    Evaluate liver segmentation model performance.
    
    Args:
        model: Trained segmentation model
        data_dir (str): Directory containing test images
        labels_dir (str): Directory containing test labels
        device: Computation device (cuda/cpu)
        batch_size (int): Batch size for evaluation
    
    Returns:
        float: Mean Dice score
    """
    # Create data dictionaries
    images = sorted([os.path.join(data_dir, img) for img in os.listdir(data_dir)])
    labels = sorted([os.path.join(labels_dir, lbl) for lbl in os.listdir(labels_dir)])
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

    # Validation transforms for original images
    val_org_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
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
        CropForegroundd(keys=["image"], source_key="image"),
    ])

    # Postprocessing transforms
    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=val_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        AsDiscreted(keys="label", to_onehot=2),
    ])

    # Create dataset and dataloader
    val_ds = Dataset(data=data_dicts, transform=val_org_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    # Initialize model
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # set up metric
    dice_metric = DiceMetric(include_background=False, reduction="mean",)

    model.load_state_dict(torch.load(os.path.join(ckpt), weights_only=True))
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result
        metric_org = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    print("Metric on original image spacing: ", metric_org)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate liver segmentation model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Directory containing test labels')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model (adjust architecture to match your trained model)
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Evaluate model
    dice_score = evaluate_model(
        model=model,
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        device=device,
        batch_size=args.batch_size
    )
    print(f"Mean Dice Score: {dice_score:.4f}")