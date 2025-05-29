from monai.networks.nets import UNet

def get_model_network(in_channels = 1, out_channels = 3, num_res_units=2, channels=(16, 32, 64, 128)):
    """
    Get the model network for training or inference.
    
    Returns:
        UNet: A 3D UNet model instance
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