from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    Spacingd, EnsureTyped, CropForegroundd,AsDiscreted
)
import matplotlib.pyplot as plt
import numpy as np


def get_transforms(mode):
    """
    Get preprocessing transforms for training, testing or inference.
    
    Args:
        mode (str): 'train', 'test', or 'inference'
    
    Returns:
        Compose: Composed transforms for the specified mode
    """
    transforms_dict = {
        'main': Compose([
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
            ),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest")
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            EnsureTyped(keys=["image", "label"])
        ]),
        'inference': Compose([
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
            ),
            Spacingd(
                keys=["image"],
                pixdim=(1.5, 1.5, 2.0),
                mode="bilinear"
            ),
            EnsureTyped(keys=["image"])
        ]),
        'post_transforms': Compose([
            AsDiscreted(keys="pred", argmax=True, to_onehot=3),
            AsDiscreted(keys="label", to_onehot=3),
        ]),
    }
    
    if mode not in transforms_dict:
        raise ValueError(f"Mode '{mode}' not supported. Choose from: {list(transforms_dict.keys())}")
    
    return transforms_dict[mode]


# Visualize preprocessed image and label
def visualize_preprocessed_image(image, label, output_path):
    image_np = image.squeeze()   # Shape: (Z, Y, X)
    label_np = label.squeeze()
    # Coronal: find the Y-slice with the most label voxels
    slice_index = np.argmax(np.sum(label_np, axis=(0, 2)))  # axis=1 is Y
    # Extract the coronal slice (Z, X)
    image_slice = image_np[:, slice_index, :]
    label_slice = label_np[:, slice_index, :]
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(image_slice, cmap='gray')
    plt.title("Coronal CT Slice")
    plt.subplot(1, 2, 2)
    plt.imshow(label_slice, cmap='gray')
    plt.title("Coronal Segmentation")
    plt.savefig(output_path)    
    plt.close()
