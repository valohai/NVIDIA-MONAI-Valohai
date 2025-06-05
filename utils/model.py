from typing import Sequence
from monai.networks.nets import UNet


def get_model_network(
    in_channels: int = 1,
    out_channels: int = 3,
    num_res_units: int = 2,
    channels: Sequence[int] = (16, 32, 64, 128)
) -> UNet:
    """
    Get the model network for training or inference.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_res_units: Number of residual units
        channels: Sequence of channel numbers

    Returns:
        A 3D UNet model instance
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=(2, 2, 2),
        num_res_units=num_res_units,
    )

    return model