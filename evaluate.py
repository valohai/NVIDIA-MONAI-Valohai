"""
Module for evaluating liver segmentation model.
"""
import argparse
import json
import os
import shutil

import torch
import valohai
from monai.data import DataLoader, Dataset, decollate_batch
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged,
                              ResizeWithPadOrCropd)

from utils.model import get_model_network
from utils.transforms import get_transforms
from utils.visualizations import plot_slices_max_label


def evaluate_model(model_path, data_dir, labels_dir, device, batch_size=1):
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
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ResizeWithPadOrCropd(
            keys=["image", "label"],
            spatial_size=(160, 160, 160)
        ),
    ])

    # Postprocessing transforms
    post_transforms = get_transforms('post_transforms')

    # Create dataset and dataloader
    val_ds = Dataset(data=data_dicts, transform=test_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    # Initialize model
    model = get_model_network()

    # set up metric
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    mean_iou_metric = MeanIoU(include_background=False, reduction="mean")

    model.load_state_dict(torch.load(os.path.join(model_path), weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_data["pred"] = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model
            )
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            val_outputs = [v.to(device) for v in val_outputs]
            val_labels = [v.to(device) for v in val_labels]

            # Plot slices with maximum label values
            plot_slices_max_label(val_inputs[0], val_labels[0], val_outputs[0], live=False)

            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            mean_iou_metric(y_pred=val_outputs, y=val_labels)

            print(json.dumps({
                "current_batch_mean_dice": dice_metric.aggregate().item(),
                "current_batch_mean_iou": mean_iou_metric.aggregate().item()
            }))

        # aggregate the final mean dice result
        metric_org = dice_metric.aggregate().item()
        metric_iou = mean_iou_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()
        mean_iou_metric.reset()

    print(json.dumps({
        "mean_dice": metric_org,
        "mean_iou": metric_iou
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate liver segmentation model')
    parser.add_argument(
        '--model_path',
        type=str,
        default='checkpoints/best_metirc_model.pth'
    )
    parser.add_argument('--batch_size', type=int, default=2)

    args = parser.parse_args()

    model = valohai.inputs('model').path(process_archives=False)
    preprocessed_data_archive = valohai.inputs('preprocessed_data').path(
        process_archives=False
    )

    # create extraction directory
    extract_dir = os.path.join(os.path.dirname(preprocessed_data_archive), "extracted_data")
    os.makedirs(extract_dir, exist_ok=True)

    # unzip the preprocessed data
    shutil.unpack_archive(preprocessed_data_archive, extract_dir, format='zip')

    # Set data directories
    data_dir = os.path.join(extract_dir, "imagesTs")
    labels_dir = os.path.join(extract_dir, "labelsTs")

    # Evaluate model
    evaluate_model(
        model_path=model,
        data_dir=data_dir,
        labels_dir=labels_dir,
        batch_size=args.batch_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
