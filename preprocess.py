"""
Module for preprocessing liver segmentation data.
"""
import os
import nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImaged, ToTensord,Spacingd,ScaleIntensityd,SaveImaged
from monai.data import Dataset, DataLoader

def preprocess_data(data_dir, labels_dir, output_dir):
    """
    Preprocess liver segmentation data with MONAI transforms.
    
    Args:
        data_dir (str): Directory containing input images
        labels_dir (str): Directory containing corresponding label masks
        output_dir (str): Directory to save processed data
        
    Returns:
        None: Saves processed images and labels to output_dir
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read dataset file paths
    images = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                    if f.endswith(('.nii', '.nii.gz', '.nrrd'))])
    labels = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) 
                    if f.endswith(('.nii', '.nii.gz', '.nrrd'))])
    
    if not images or not labels:
        raise ValueError("No valid image or label files found in the specified directories")
    
    # Create list of dictionaries with image-label pairs
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
    
    # Define preprocessing transforms
    transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.5, 1.5, 2.0), 
            mode=("bilinear", "nearest"),
            align_corners=[True, None]
        ),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        SaveImaged(
            keys=["image", "label"], 
            output_dir=output_dir, 
            output_postfix="",
            output_ext=".nii.gz",
            separate_folder=False,
            resample=False
        )
    ])
    
    # Create dataset and dataloader
    dataset = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(dataset, batch_size=1, num_workers=2)
    
    # Process and save all images
    for _ in loader:
        pass  # Processing happens through SaveImaged transform

# call the file
# if __name__ == "__main__":
#     preprocess_data("liver_dataset", "liver_dataset", "processed_data")
