"""
Module for preprocessing liver segmentation data, including training and test sets.
"""
import os
import argparse
import numpy as np
from glob import glob
import nibabel as nib

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    Spacingd, ResizeWithPadOrCropd, SaveImaged, Lambdad
)
from monai.data import Dataset
from monai.utils import set_determinism

FILE_KEYS = ["image", "label"]

def preprocess_train_val(data_dir, labels_dir, output_dir, check_sample=False):
    images = sorted(glob(os.path.join(data_dir, '*.nii*')))
    labels = sorted(glob(os.path.join(labels_dir, '*.nii*')))

    if not images or not labels:
        raise ValueError("No valid image or label files found.")

    os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)

    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]


    preprocess = Compose([
        LoadImaged(keys=FILE_KEYS),
        EnsureChannelFirstd(keys=FILE_KEYS),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
        ),
        Spacingd(
            keys=FILE_KEYS,
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest")
        ),
        Lambdad(keys="label", func=lambda x: np.where(x == 2, 1, x)),
        ResizeWithPadOrCropd(keys=FILE_KEYS, spatial_size=(160, 160, 160)),
    ])

    dataset = Dataset(data=data_dicts, transform=preprocess)


    for i, sample in enumerate(dataset):
        base_name = os.path.splitext(os.path.basename(data_dicts[i]["image"]))[0]

        image = sample["image"].detach().cpu().numpy().squeeze()
        label = sample["label"].detach().cpu().numpy().squeeze()

        nib.save(nib.Nifti1Image(image, np.eye(4)), os.path.join(output_dir, "imagesTr", f"{base_name}.gz"))
        nib.save(nib.Nifti1Image(label, np.eye(4)), os.path.join(output_dir, "labelsTr", f"{base_name}.gz"))

def preprocess_test(test_dir, output_dir):
    images = sorted(glob(os.path.join(test_dir, '*.nii*')))
    if not images:
        raise ValueError("No valid test images found.")

    os.makedirs(os.path.join(output_dir, "imagesTs"), exist_ok=True)
    data_dicts = [{"image": img} for img in images]

    preprocess = Compose([
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
        ResizeWithPadOrCropd(keys=["image"], spatial_size=(160, 160, 160)),
    ])

    dataset = Dataset(data=data_dicts, transform=preprocess)

    for sample in dataset:
        base_name = os.path.splitext(os.path.basename(data_dicts["image"]))[0]
        image = sample["image"].detach().cpu().numpy().squeeze()
        nib.save(nib.Nifti1Image(image, np.eye(4)), os.path.join(output_dir, "imagesTs", f"{base_name}.gz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess liver dataset (Train/Val + Test)")
    parser.add_argument('--data_dir', type=str, default='-Valohai-MONAI-Medical-Imaging-/data/imagesTr',
                        help='Directory with training images')
    parser.add_argument('--labels_dir', type=str, default='-Valohai-MONAI-Medical-Imaging-/data/labelsTr',
                        help='Directory with training labels')
    parser.add_argument('--test_dir', type=str, default='-Valohai-MONAI-Medical-Imaging-/data/imagesTs',
                        help='Directory with test images')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Directory to save preprocessed data')
    parser.add_argument('--check_sample', type=bool, default=False,
                        help='Visualize a sample image+label pair')

    args = parser.parse_args()

    set_determinism(seed=0)

    preprocess_train_val(
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        check_sample=args.check_sample
    )

    preprocess_test(
        test_dir=args.test_dir,
        output_dir=args.output_dir
    )
