from monai.networks.nets import UNet

def get_model_network():
    """
    Get the model network for training or inference.
    
    Returns:
        UNet: A 3D UNet model instance
    """
    
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="batch",  # Add normalization for better training stability
    )
    
    return model