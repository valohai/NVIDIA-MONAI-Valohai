"""
Module for training liver segmentation model.
"""
import os
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose
from monai.networks.layers import Norm
import argparse
from sklearn.model_selection import train_test_split
from monai.data import DataLoader, Dataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd,
    RandRotate90d, RandFlipd, RandGaussianNoised, RandCropByPosNegLabeld
)
from utils.model import get_model_network
from utils.transforms import get_transforms
from utils.visualizations import plot_slices_max_label
import valohai
import shutil
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Train liver segmentation model')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ckpt',type=str,default='checkpoints')
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--num_res_units', type=int, default=2)
    parser.add_argument('--channels', type=lambda s: list(map(int, s.split(','))))
    return parser.parse_args()


def train_model(train_loader, val_loader, model, num_epochs=100, learning_rate=1e-4,ckpt_path="checkpoints"):
    """
    Train liver segmentation model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization
        ckpt_path (str): Path to save checkpoints
        model: Model instance to train
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(ckpt_path, exist_ok=True)

    model = model.to(device)


    # Loss function and optimizer
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    
    # Metric
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    mean_iou_metric = MeanIoU(include_background=False, reduction="mean")
    
    # Training loop
    best_dice_score = -1
    best_dice_epoch = -1
    epoch_loss_values = []
    dice_values = []
    
    post_transforms = get_transforms('post_transforms')

    
    for epoch in range(num_epochs):
        print(json.dumps({
            "epochs": epoch + 1,
        }))
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}, val_loss: {epoch_loss / step:.4f}", end='\r')
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print(json.dumps({
            "epoch": epoch + 1,
            "loss": epoch_loss
        }))
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    
                    # Sliding window inference for large images
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    # Create a batch dictionary and apply post transforms
                    val_batch_data = [{"pred": pred, "label": label} for pred, label 
                                     in zip(decollate_batch(val_outputs), decollate_batch(val_labels))]
                    
                    # Apply post-transforms to the batch data
                    val_batch_data = [post_transforms(d) for d in val_batch_data]
                    
                    # Extract processed predictions and labels for metric computation
                    val_outputs = [d["pred"] for d in val_batch_data]
                    val_labels = [d["label"] for d in val_batch_data]

                    plot_slices_max_label(val_inputs[0], val_labels[0], val_outputs[0], output_dir='/valohai/outputs/train_progress/')

                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)  
                    mean_iou_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice                    
                dice_score = dice_metric.aggregate().item()
                mean_iou_score = mean_iou_metric.aggregate().item()
                dice_metric.reset()
                mean_iou_metric.reset()

                dice_values.append(dice_score)
                if dice_score > best_dice_score:
                    best_dice_score = dice_score
                    best_dice_epoch = epoch + 1

                    #Save model
                    model_output_path = valohai.outputs().path('model.pth')
                    torch.save(model.state_dict(), model_output_path)

                    # Write metadata after model file exists
                    file_metadata = {
                        "valohai.alias": "latest-model"
                    }
                    with open(f"{model_output_path}.metadata.json", "w") as f:
                        json.dump(file_metadata, f)
                                    
                # valohai metadata
                print(json.dumps({
                    "epoch": epoch + 1,
                    "val_dice": dice_score,
                    "val_mean_iou": mean_iou_score,
                    "best_dice_score": best_dice_score,
                    "best_dice_epoch": best_dice_epoch
                }))
    
    print(f"Training completed, best dice_score: {best_dice_score:.4f} at epoch: {best_dice_epoch}")

    return model

def get_data_loaders(data_dir, labels_dir, batch_size=2, val_split=0.2):
    """
    Get data loaders for training and validation datasets.
    
    Args:
        data_dir (str): Directory containing input images
        labels_dir (str): Directory containing label masks
        batch_size (int): Batch size for training
        val_split (float): Fraction of data to use for validation
    
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
    """
    images = sorted(os.listdir(data_dir))
    labels = sorted(os.listdir(labels_dir))

    data_dicts = [{"image": os.path.join(data_dir, img), "label": os.path.join(labels_dir, lbl)}
                  for img, lbl in zip(images, labels)]

    train_data, val_data = train_test_split(data_dicts, test_size=val_split, random_state=42)

    # Define Random transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(160, 160, 160),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
            allow_smaller=True
        ),
        RandRotate90d(keys=["image", "label"], prob=0.5),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
        RandGaussianNoised(keys=["image"], prob=0.5)
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
    ])

    train_loader = DataLoader(
        Dataset(data=train_data, transform=train_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    val_loader = DataLoader(
        Dataset(data=val_data, transform=val_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_loader, val_loader

if __name__ == "__main__":
    
    args = parse_args()
    


    preprocessed_data_archive = valohai.inputs('preprocessed_data').path(process_archives=False)

    # create extraction directory
    extract_dir = os.path.join(os.path.dirname(preprocessed_data_archive), "extracted_data")
    os.makedirs(extract_dir, exist_ok=True)

    #unzip the preprocessed data
    shutil.unpack_archive(preprocessed_data_archive, extract_dir, format='zip')

    # Set data directories
    data_dir = os.path.join(extract_dir, "imagesTr")
    labels_dir = os.path.join(extract_dir, "labelsTr")



    train_loader, val_loader = get_data_loaders(data_dir,labels_dir, args.batch_size)

    # Initialize Model
    init_model = get_model_network(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        num_res_units=args.num_res_units,
        channels=args.channels
    )

    # Train model
    model = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=init_model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        ckpt_path=args.ckpt,
    )