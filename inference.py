"""
Module for running inference with liver segmentation model.
"""
import torch
import nibabel as nib
from monai.transforms import Compose, LoadImaged, ToTensord

def run_inference(model, input_image_path):
    """
    Run inference on a single liver image.
    Args:
        model: Trained segmentation model
        input_image_path (str): Path to input image
    """
    # To be implemented
    pass
