"""
Module for preprocessing liver segmentation data.
"""
import os
import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    ScaleIntensityd, SaveImaged, EnsureTyped,ResizeWithPadOrCropd
)
from monai.data import Dataset, DataLoader, CacheDataset
from sklearn.model_selection import train_test_split
import argparse

def preprocess_data(data_dir, labels_dir, output_dir,batch_size):
    """
    Preprocess liver segmentation data, split into train/val, and create loaders.
    
    Args:
        data_dir (str): Directory containing input images
        labels_dir (str): Directory containing corresponding label masks
        output_dir (str): Directory to save processed data
        
    Returns:
        train_loader, val_loader: MONAI DataLoaders for training and validation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read dataset file paths
    images = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                     if f.endswith(('.nii', '.nii.gz', '.nrrd'))])
    labels = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) 
                     if f.endswith(('.nii', '.nii.gz', '.nrrd'))])
    
    if not images or not labels:
        raise ValueError("No valid image or label files found in the specified directories")
    
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
    
    # Train/Val split (80/20)
    train_files, val_files = train_test_split(data_dicts, test_size=0.2, random_state=42)
    
    # Define preprocessing transforms
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.5, 1.5, 2.0), 
            mode=("bilinear", "nearest"),
            align_corners=[True, None]
        ),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            # Resize or pad to a fixed shape
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(160, 160, 160)),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Use CacheDataset for efficiency if memory allows
    train_ds = Dataset(data=train_files, transform=transforms)
    val_ds = Dataset(data=val_files, transform=transforms,)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Run preprocessing (to trigger SaveImaged)
    print("Processing and saving training data...")
    for _ in train_loader:
        # check one sampe data
        sample = train_loader.dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        break
    print("Processing and saving validation data...")
    for _ in val_loader:
        # check one sampe data
        sample = val_loader.dataset[0]
        print(f"Image shape val:")
        print(f"Image shape: {sample['image'].shape}")
    
        break
    
    return train_loader, val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess liver dataset')
    parser.add_argument('--data_dir', type=str, default='liver_dataset/images_Tr',
                        help='Directory containing input images')
    parser.add_argument('--labels_dir', type=str, default='liver_dataset/labels_Tr',
                        help='Directory containing label masks')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Directory to save processed data')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for data loading')
    
    args = parser.parse_args()
    
    # Call the preprocessing function with command line arguments
    train_loader, val_loader = preprocess_data(
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )