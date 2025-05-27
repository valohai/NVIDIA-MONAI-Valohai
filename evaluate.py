"""
Module for evaluating liver segmentation model.
"""
import os
import torch
import argparse
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
)
from monai.handlers.utils import from_engine
import matplotlib.pyplot as plt
from utils.transforms import get_transforms
from utils.model import get_model_network
import valohai
import shutil

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
    ])

    # Postprocessing transforms
    post_transforms = get_transforms('post_transforms')

    # Create dataset and dataloader
    val_ds = Dataset(data=data_dicts, transform=test_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    # Initialize model
    model = get_model_network()

    # set up metric
    dice_metric = DiceMetric(include_background=False, reduction="mean",)

    model.load_state_dict(torch.load(os.path.join(model_path), weights_only=True))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            val_outputs = [v.to(device) for v in val_outputs]
            val_labels = [v.to(device) for v in val_labels]
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)

            print(f"Processed {len(val_outputs)} images in current batch.")
            #print metric for current batch
            print("Current batch mean dice: ", dice_metric.aggregate().item())
        

        # aggregate the final mean dice result
        metric_org = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    print("Metric on original image spacing: ", metric_org)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate liver segmentation model')
    parser.add_argument('--data_dir', type=str, default='-Valohai-MONAI-Medical-Imaging-/processed_data/imagesTs')
    parser.add_argument('--labels_dir', type=str, default='-Valohai-MONAI-Medical-Imaging-/processed_data/labelsTs')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_metirc_model.pth')
    parser.add_argument('--batch_size', type=int, default=1)
    
    args = parser.parse_args()

    model = valohai.inputs('model').path(process_archives=False)

    preprocessed_data_archive = valohai.inputs('preprocessed_data').path(process_archives=False)

    # create extraction directory
    extract_dir = os.path.join(os.path.dirname(preprocessed_data_archive), "extracted_data")
    os.makedirs(extract_dir, exist_ok=True)

    #unzip the preprocessed data
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
