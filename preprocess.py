"""
Module for preprocessing liver segmentation data, including training and test sets.
"""
import os
import argparse
import numpy as np
from glob import glob
import nibabel as nib
from monai.data import Dataset
from monai.utils import set_determinism
from utils import get_transforms

FILE_KEYS = ["image", "label"]

def preprocess_train_val(data_dir, labels_tr, test_dir, labels_ts, output_dir, check_sample=False):
    # Process training data
    train_images = sorted(glob(os.path.join(data_dir, '*.nii*')))
    train_labels = sorted(glob(os.path.join(labels_tr, '*.nii*')))

    if not train_images or not train_labels:
        raise ValueError("No valid training image or label files found.")

    # Process test data
    test_images = sorted(glob(os.path.join(test_dir, '*.nii*')))
    test_labels = sorted(glob(os.path.join(labels_ts, '*.nii*')))

    if not test_images or not test_labels:
        raise ValueError("No valid test image or label files found.")

    # Create output directories
    os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imagesTs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTs"), exist_ok=True)

    # Create data dictionaries
    train_data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]
    test_data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(test_images, test_labels)]

    preprocess_train = get_transforms('train')
    


    # Process training data
    train_dataset = Dataset(data=train_data_dicts, transform=preprocess_train)
    for i, sample in enumerate(train_dataset):
        base_name = os.path.splitext(os.path.basename(train_data_dicts[i]["image"]))[0]

        image = sample["image"].detach().cpu().numpy().squeeze()
        label = sample["label"].detach().cpu().numpy().squeeze().astype(np.int16)

        # Use affine from MONAI transform metadata
        image_affine = sample["image_meta_dict"]["affine"]
        label_affine = sample["label_meta_dict"]["affine"]

        nib.save(nib.Nifti1Image(image, image_affine), os.path.join(output_dir, "imagesTr", f"{base_name}.gz"))
        nib.save(nib.Nifti1Image(label, label_affine), os.path.join(output_dir, "labelsTr", f"{base_name}.gz"))

    preprocess_test = get_transforms('test')
    # Process test data
    test_dataset = Dataset(data=test_data_dicts, transform=preprocess_test)
    for i, sample in enumerate(test_dataset):
        base_name = os.path.splitext(os.path.basename(test_data_dicts[i]["image"]))[0]

        image = sample["image"].detach().cpu().numpy().squeeze()
        label = sample["label"].detach().cpu().numpy().squeeze().astype(np.int16)

        # Use affine from MONAI transform metadata
        image_affine = sample["image_meta_dict"]["affine"]
        label_affine = sample["label_meta_dict"]["affine"]

        nib.save(nib.Nifti1Image(image, image_affine), os.path.join(output_dir, "imagesTs", f"{base_name}.gz"))
        nib.save(nib.Nifti1Image(label, label_affine), os.path.join(output_dir, "labelsTs", f"{base_name}.gz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess liver dataset (Train/Val + Test)")
    parser.add_argument('--data_dir', type=str, default='-Valohai-MONAI-Medical-Imaging-/data/imagesTr')
    parser.add_argument('--labels_tr', type=str, default='-Valohai-MONAI-Medical-Imaging-/data/labelsTr')
    parser.add_argument('--labels_ts', type=str, default='-Valohai-MONAI-Medical-Imaging-/data/labelsTs')
    parser.add_argument('--test_dir', type=str, default='-Valohai-MONAI-Medical-Imaging-/data/imagesTs')
    parser.add_argument('--output_dir', type=str, default='-Valohai-MONAI-Medical-Imaging-/processed_data')
    parser.add_argument('--check_sample', type=bool, default=False)


    args = parser.parse_args()

    set_determinism(seed=0)

    preprocess_train_val(
        data_dir=args.data_dir,
        labels_tr=args.labels_tr,
        test_dir = args.test_dir,
        labels_ts = args.labels_ts,
        output_dir=args.output_dir,
        check_sample=args.check_sample
    )