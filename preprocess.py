"""
Module for preprocessing liver segmentation data.
"""
import os
import nibabel as nib
import numpy as np
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    ResizeWithPadOrCropd,
    Lambdad
)
from monai.data import Dataset, DataLoader, PersistentDataset
from sklearn.model_selection import train_test_split
from monai.utils import first, set_determinism
import argparse
import matplotlib.pyplot as plt

def preprocess_data(data_dir, labels_dir, output_dir,batch_size,check_sample=False):   
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

    set_determinism(seed=0)
    
    # Train/Val split (80/20)
    train_files, val_files = train_test_split(data_dicts, test_size=0.2, random_state=42)
    
    # Define preprocessing transforms
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            # apply padding to make sure the image and label have the same size
            Lambdad(keys="label", func=lambda x: np.where(x == 2, 1, x)),  # Map value 2 to 1

            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(160, 160, 160)),  # Resize to a fixed size

            


            # user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(96, 96, 96),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
        ]
    )

    # check transforms in dataloader
    if check_sample:
        check_ds = Dataset(data=val_files, transform=train_transforms)
        check_loader = DataLoader(check_ds, batch_size=1)
        check_data = first(check_loader)
        image, label = (check_data["image"][0][0], check_data["label"][0][0])
        print(f"image shape: {image.shape}, label shape: {label.shape}")
        
        # Dynamically select the middle slice of the third dimension
        slice_idx = 140
        print(f"Plotting slice index: {slice_idx}")
        
        # Plot the image and label
        plt.figure("check", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(image[:, :, slice_idx], cmap="gray")
        plt.axis("off")  # Remove axes for better visualization
        plt.subplot(1, 2, 2)
        plt.title("Label")
        plt.imshow(label[:, :, slice_idx], cmap="jet")  # Use a colormap for labels
        plt.axis("off")  # Remove axes for better visualization
        plt.show()
    
    # Use CacheDataset for efficiency if memory allows
    # train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)

    cache_dir = '-Valohai-MONAI-Medical-Imaging-\preprocessed_data'

    dataset = PersistentDataset(data=data_dicts, transform=train_transforms, cache_dir=cache_dir)

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
    parser.add_argument('--check_sample', type=bool, default=True,
                        help='Check sample data after preprocessing')
    
    args = parser.parse_args()
    
    # Call the preprocessing function with command line arguments
    train_loader, val_loader = preprocess_data(
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        check_sample=args.check_sample
    )