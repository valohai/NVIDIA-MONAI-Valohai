"""
Module for training liver segmentation model.
"""
import os
import torch
from monai.networks.nets import SwinUNETR,UNet  # Changed from UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose
from monai.networks.layers import Norm


import argparse
from preprocess import preprocess_data


def train_model(train_loader, val_loader, num_epochs=100, learning_rate=1e-4,ckpt_path="checkpoints"):
    """
    Train liver segmentation model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization
        ckpt_path (str): Path to save checkpoints
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(ckpt_path, exist_ok=True)



    # Initialize model
    # model = SwinUNETR(
    #     img_size=(160, 160, 160),
    #     in_channels=1,
    #     out_channels=2,
    #     feature_size=48,        # Base feature size
    #     drop_rate=0.0,         # Dropout rate
    #     attn_drop_rate=0.0,    # Attention dropout rate
    #     dropout_path_rate=0.0, # Drop path rate
    #     use_checkpoint=False    # Use gradient checkpointing to save memory
    # ).to(device)

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)


    # Loss function and optimizer
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    
    # Metric
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # post processing transforms
    
    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
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
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
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
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)  

                # aggregate the final mean dice                    
                metric = dice_metric.aggregate().item()
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(ckpt_path, "best_metirc_model.pth"))
                    print("saved new best metric model")
                    
                print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                      f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    print(f"Training completed, best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train liver segmentation model')
    parser.add_argument('--data_dir', type=str, default='liver_dataset/images_Tr',
                        help='Directory containing input images')
    parser.add_argument('--labels_dir', type=str, default='liver_dataset/labels_Tr',
                        help='Directory containing label masks')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Directory to save processed data')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--ckpt',type=str,default='checkpoints',help='checkpoint directory')
    
    args = parser.parse_args()
    
    # Get data loaders
    train_loader, val_loader = preprocess_data(
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    # Train model
    model = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        ckpt_path=args.ckpt
    )