from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    Spacingd, EnsureTyped, CropForegroundd, ResizeWithPadOrCropd
)

def get_transforms(mode):
    """
    Get preprocessing transforms for training, testing or inference.
    
    Args:
        mode (str): 'train', 'test', or 'inference'
    
    Returns:
        Compose: Composed transforms for the specified mode
    """
    
    transforms_dict = {
        'train': Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
            ),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest")
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # reszie to fixed size 160
            ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(160, 160, 160),
            ),
            EnsureTyped(keys=["image", "label"])
        ]),
        
        'test': Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
            ),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest")
            ),
            EnsureTyped(keys=["image", "label"])
        ]),
        
        'inference': Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
            ),
            Spacingd(
                keys=["image"],
                pixdim=(1.5, 1.5, 2.0),
                mode="bilinear"
            ),
            CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
            EnsureTyped(keys=["image"])
        ])
    }
    
    if mode not in transforms_dict:
        raise ValueError(f"Mode '{mode}' not supported. Choose from: {list(transforms_dict.keys())}")
    
    return transforms_dict[mode]