
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from utils.visualizations import visualize_preprocessed_image


FILE_KEYS = ["image", "label"]


def process_dataset(data_dicts, dataset_transform, output_subdir, output_dir):
    """
    Process a dataset with transforms and save the results.
    
    Args:
        data_dicts (list): List of dictionaries with image and label paths
        dataset_transform: MONAI transforms to apply
        output_subdir (str): Subdirectory name for images/labels (e.g., 'imagesTr')
        output_dir (str): Base output directory
    """
    dataset = Dataset(data=data_dicts, transform=dataset_transform)
    
    # Create output directories if they don't exist
    images_dir = os.path.join(output_dir, output_subdir)
    labels_dir = os.path.join(output_dir, output_subdir.replace('images', 'labels'))
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"Processing {len(dataset)} samples for {output_subdir}...")
    for i, sample in enumerate(tqdm(dataset, desc=f"Processing {output_subdir}", unit="sample")):
        base_name = os.path.splitext(os.path.basename(data_dicts[i]["image"]))[0]

        image = sample["image"].detach().cpu().numpy().squeeze()
        label = sample["label"].detach().cpu().numpy().squeeze().astype(np.int16)

        # Use affine from MONAI transform metadata
        image_affine = sample["image_meta_dict"]["affine"]
        label_affine = sample["label_meta_dict"]["affine"]

        # Save the processed files
        nib.save(nib.Nifti1Image(image, image_affine), os.path.join(images_dir, f"{base_name}.gz"))
        nib.save(nib.Nifti1Image(label, label_affine), os.path.join(labels_dir, f"{base_name}.gz"))

        if i < 5:  # Visualize only the first 5 samples
            visualize_preprocessed_image(image, label, f"/valohai/outputs/sample_{i}.png")

            
    
    print(f"Saved {len(dataset)} samples to {images_dir} and {labels_dir}")

def preprocess_train_val(data_dir, labels_tr, output_dir):    
    # Process training and test data
    volumes = sorted(glob(os.path.join(data_dir, '*.nii*')))
    masks = sorted(glob(os.path.join(labels_tr, '*.nii*')))

    if not volumes or not masks:
        raise ValueError("No valid training image or label files found.")

    # spit train_images to train and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        volumes, masks, test_size=0.1, random_state=42
    )
    # Create output directories
    os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imagesTs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTs"), exist_ok=True)

    # Create data dictionaries
    train_data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]
    test_data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(test_images, test_labels)]

    # Process training data
    process_dataset(
        data_dicts=train_data_dicts,
        dataset_transform=get_transforms('main'),
        output_subdir="imagesTr",
        output_dir=output_dir
    )
    
    # Process test data
    process_dataset(
        data_dicts=test_data_dicts,
        dataset_transform=get_transforms('main'),
        output_subdir="imagesTs",
        output_dir=output_dir
    )

    # Get zip output path
    zip_output_path = valohai.outputs().path("preprocessed")

    # Zip the entire processed output folder
    shutil.make_archive(zip_output_path, 'zip', output_dir)

    # Save Valohai metadata
    metadata = {
        "preprocessed.zip": {
            "valohai.dataset-versions": [
                 "dataset://task03_liver/version1"
             ],
        }
    }

    metadata_path = valohai.outputs().path("valohai.metadata.jsonl")
    with open(metadata_path, "w") as outfile:
        for file_name, file_metadata in metadata.items():
            json.dump({"file": file_name, "metadata": file_metadata}, outfile)
            outfile.write("\n")

if __name__ == "__main__":

    # Get the dataset .rar file path from Valohai
    dataset_archive = valohai.inputs('dataset').path(process_archives=False)


    # Create extraction directory
    extract_dir = os.path.join(os.path.dirname(dataset_archive), "extracted_data")
    os.makedirs(extract_dir, exist_ok=True)
    
    # Unpack the dataset
    if not os.path.exists(dataset_archive):
        raise ValueError(f"Dataset archive {dataset_archive} does not exist.")
    
    print(f"Extracting {dataset_archive} to {extract_dir}")

    #Check zip or tar
    if dataset_archive.endswith('.tar'):
        shutil.unpack_archive(dataset_archive, extract_dir=extract_dir, format='tar')
    elif dataset_archive.endswith('.zip'):
        shutil.unpack_archive(dataset_archive, extract_dir=extract_dir, format='zip')

    
    # Set paths to the extracted data folders
    # Find paths to required directories
    imagesTr_path = glob(os.path.join(extract_dir, "**", "imagesTr"), recursive=True)
    labelsTr_path = glob(os.path.join(extract_dir, "**", "labelsTr"), recursive=True)


    if not imagesTr_path or not labelsTr_path:
        raise FileNotFoundError("imagesTr or labelsTr folder not found in extracted dataset.")

    # Create output directory
    output_dir = os.path.join(os.getcwd(), "processed_data")
    os.makedirs(output_dir, exist_ok=True)
    
    set_determinism(seed=0)
    

    preprocess_train_val(
        data_dir=imagesTr_path[0],
        labels_tr=labelsTr_path[0],
        output_dir=output_dir
    )