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
from utils.transforms import get_transforms
import shutil
import valohai

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

    #Get output file and folder path

    zip_output_path = valohai.outputs().path('/valohai/outputs/preprocessed')
    shutil.make_archive(zip_output_path, 'zip', output_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess liver dataset (Train/Val + Test)")
    parser.add_argument('--output_dir', type=str, default='processed_data')
    parser.add_argument('--check_sample', type=bool, default=False)
    args = parser.parse_args()



    # Get the dataset .rar file path from Valohai
    dataset_archive = valohai.inputs('dataset').path(process_archives=False)
    
    # Create extraction directory
    extract_dir = os.path.join(os.path.dirname(dataset_archive), "extracted_data")
    os.makedirs(extract_dir, exist_ok=True)
    
    # Unpack the dataset
    if not os.path.exists(dataset_archive):
        raise ValueError(f"Dataset archive {dataset_archive} does not exist.")
    
    print(f"Extracting {dataset_archive} to {extract_dir}")
    shutil.unpack_archive(dataset_archive, extract_dir=extract_dir, format='zip')
    
    # Set paths to the extracted data folders
    data_dir = os.path.join(extract_dir, "data_min", "imagesTr")
    labels_tr = os.path.join(extract_dir, "data_min", "labelsTr")
    test_dir = os.path.join(extract_dir, "data_min", "imagesTs")
    labels_ts = os.path.join(extract_dir, "data_min", "labelsTs")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    set_determinism(seed=0)
    
    print(f"Processing data from {data_dir}")
    print(f"Training labels from {labels_tr}")
    print(f"Test data from {test_dir}")
    print(f"Test labels from {labels_ts}")


    set_determinism(seed=0)

    preprocess_train_val(
        data_dir=data_dir,
        labels_tr=labels_tr,
        test_dir = test_dir,
        labels_ts = labels_ts,
        output_dir=output_dir,
        check_sample=args.check_sample
    )