from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    Spacingd, EnsureTyped, CropForegroundd, ResizeWithPadOrCropd,AsDiscreted, Invertd, SaveImaged
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