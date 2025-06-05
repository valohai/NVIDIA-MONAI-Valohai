
from typing import Dict, Literal
from monai.transforms import (AsDiscreted, Compose, EnsureChannelFirstd,
                              EnsureTyped, LoadImaged, Resized,
                              ScaleIntensityRanged, Spacingd)

TransformMode = Literal['main', 'inference', 'post_transforms']


def get_transforms(mode: TransformMode) -> Compose:
    """
    Get preprocessing transforms for training, testing or inference.
    
    Args:
        mode: Transform mode to use ('main', 'inference', or 'post_transforms')
    
    Returns:
        Composed transforms for the specified mode

    Raises:
        ValueError: If mode is not one of the supported values
    """
    transforms_dict: Dict[TransformMode, Compose] = {
        'main': Compose([
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest")
            ),
            EnsureTyped(keys=["image", "label"]),
            Resized(
                keys=["image", "label"],
                spatial_size=(160, 160, 160),
            )
        ]),
        'inference': Compose([
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            Spacingd(
                keys=["image"],
                pixdim=(1.5, 1.5, 2.0),
                mode="bilinear"
            ),
            EnsureTyped(keys=["image"]),
            Resized(
                keys=["image"],
                spatial_size=(160, 160, 160),
            )
        ]),
        'post_transforms': Compose([
            AsDiscreted(keys="pred", argmax=True, to_onehot=3),
            AsDiscreted(keys="label", to_onehot=3),
        ]),
    }
    if mode not in transforms_dict:
        raise ValueError(
            f"Mode '{mode}' not supported. Choose from: {list(transforms_dict.keys())}"
        )
    
    return transforms_dict[mode]


