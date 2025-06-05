from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import valohai


def visualize_preprocessed_image(
    image: np.ndarray,
    label: np.ndarray,
    output_path
) -> None:
    """
    Visualize preprocessed image and label.

    Args:
        image: Input image array
        label: Label array
        output_path: Path to save the visualization
    """
    image_np = image.squeeze()  # Shape: (Z, Y, X)
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


def plot_slices_max_label(
    input_tensor: Union[torch.Tensor, np.ndarray],
    label_tensor: Union[torch.Tensor, np.ndarray],
    pred_tensor: Union[torch.Tensor, np.ndarray],
    output_dir=valohai.outputs("my-output"),
    live: bool = True
) -> None:
    """
    Plot slices with maximum label values.

    Args:
        input_tensor: Input image tensor
        label_tensor: Ground truth label tensor
        pred_tensor: Prediction tensor
        output_dir: Directory to save the plots
        live: Whether to upload plots live to Valohai
    """
    # Convert to numpy and squeeze channel if necessary
    input_np = (
        input_tensor.cpu().numpy().squeeze()
        if isinstance(input_tensor, torch.Tensor)
        else input_tensor.squeeze()
    )
    label_np = (
        label_tensor.cpu().numpy()
        if isinstance(label_tensor, torch.Tensor)
        else label_tensor
    )
    pred_np = (
        pred_tensor.cpu().numpy()
        if isinstance(pred_tensor, torch.Tensor)
        else pred_tensor
    )

    # Convert one-hot to single-channel mask
    if label_np.shape[0] == 3:
        label_np = np.argmax(label_np, axis=0)
    if pred_np.shape[0] == 3:
        pred_np = np.argmax(pred_np, axis=0)

    # Find slice with the most label voxels (in Z axis)
    sums = np.sum(label_np, axis=(0, 1))  # over X and Y
    slice_idx = np.argmax(sums)

    input_slice = input_np[:, :, slice_idx]
    label_slice = label_np[:, :, slice_idx]
    pred_slice = pred_np[:, :, slice_idx]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(input_slice, cmap="gray")
    axs[0].set_title("Input")
    axs[1].imshow(label_slice, cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred_slice, cmap="gray")
    axs[2].set_title("Prediction")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    save_path = output_dir.path(f"max_label_slice_{slice_idx}.png")
    plt.savefig(save_path)
    if live:
        valohai.outputs().live_upload(save_path)
    plt.close()