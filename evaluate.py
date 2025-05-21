"""
Module for evaluating liver segmentation model.
"""
import os
import torch
import argparse
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from preprocess import preprocess_data

def evaluate_model(model, data_loader, device):
    """
    Evaluate liver segmentation model performance.
    
    Args:
        model: Trained segmentation model
        data_loader (DataLoader): Evaluation data loader
        device: Computation device (cuda/cpu)
    
    Returns:
        float: Mean Dice score
    """
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_trans = AsDiscrete(threshold=0.5)
    
    with torch.no_grad():
        metric_sum = 0.0
        metric_count = 0
        
        for batch_data in data_loader:
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device)
            )
            
            # Inference with sliding window
            roi_size = (160, 160, 160)
            outputs = sliding_window_inference(
                inputs, roi_size, 4, model, overlap=0.5
            )
            
            # Post-process outputs
            outputs = post_trans(outputs)
            dice_metric(y_pred=outputs, y=labels)
            
        # Aggregate metrics
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        
        return metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate liver segmentation model')
    parser.add_argument('--data_dir', type=str, default='liver_dataset/images_Ts',
                        help='Directory containing test images')
    parser.add_argument('--labels_dir', type=str, 
                        help='Directory containing test labels',default='liver_dataset/labels_Ts')
    parser.add_argument('--model_path', type=str,default='checkpoints/best_model_full.pth',
                        help='Path to saved model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model from checkpoint
    # Load full model directly
    from monai.networks.nets import UNet  # or SwinUNETR if you used it during training
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("Model weights loaded successfully.")  
    model.to(device)

    
    # Get test data loader
    _, test_loader = preprocess_data(
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        batch_size=args.batch_size
    )
    
    # Evaluate model
    dice_score = evaluate_model(model, test_loader, device)
    print(f"Mean Dice Score: {dice_score:.4f}")