# MONAI Training Pipeline

This repository contains a complete pipeline for training a UNet model for liver segmentation using the **MONAI Task03\_Liver** dataset, orchestrated with **Valohai**.

## Overview

This project demonstrates how to:

1. Download and preprocess the Task03\_Liver dataset
2. Train a UNet model using MONAI
3. Evaluate model performance using Dice Score
4. Run inference on unseen data

## Project Structure

* `preprocess.py`: Downloads and preprocesses the Task03\_Liver data
* `train.py`: Trains the UNet model using MONAI
* `evaluate.py`: Evaluates model performance on the test set
* `inference.py`: Performs segmentation inference on new images
* `valohai.yaml`: Defines the Valohai pipeline and execution steps
* `requirements.txt`: Core Python dependencies

## Pipeline Steps
![image](https://github.com/user-attachments/assets/bdf63043-771e-41b6-b250-6c85b2cd013c)

This pipeline automates the full workflow for medical image segmentation using a U-Net architecture on the Task03\_Liver dataset from the Medical Segmentation Decathlon.

### 1. **Preprocess Dataset**

* Downloads the Task03\_Liver dataset from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/).
* Applies preprocessing transforms such as resampling, cropping, and resizing.
* Splits the dataset into training, validation, and test sets, and generates manifest files.
* Saves the preprocessed volumes and labels as zip using Datasets (check: https://docs.valohai.com/hc/en-us/articles/18704302494481-Creating-datasets)


### 2. **Train Model**

* Trains a configurable U-Net model on the preprocessed dataset.
* Supports custom settings for:

  * Number of epochs
  * Learning rate
  * Batch size
  * Input/output channels
  * Number of residual units
  * Channel depth at each stage

### 3. **Evaluate Model**

* Evaluates model performance using metrics such as:

  * Dice Similarity Coefficient (DSC)
  * Intersection over Union (IoU)
* Produces detailed logs and segmentation quality visualizations.

### 4. **Run Inference**

* Performs sliding window inference on new or unseen volumetric data.
* Input:
  * Raw volume in NIfTI format (`.nii.gz`)

* Outputs:

  * Prediction masks in NIfTI format (`.nii.gz`)
  * Visualization snapshots for qualitative analysis

### Run the pipeline
vh pipeline run train_and_evaluate

## Dataset

This project uses the **Task03\_Liver** dataset from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/). The dataset contains CT volumes of the liver with manual segmentation masks.

## Model

The training pipeline uses the **UNet** architecture implemented in [MONAI](https://monai.io/), optimized for medical image segmentation tasks.

## License

This project leverages the MONAI framework and uses datasets from the Medical Segmentation Decathlon. It is distributed under the Apache License 2.0.

## Acknowledgments

* [Project MONAI](https://github.com/Project-MONAI/MONAI)
* [Valohai](https://valohai.com/) for orchestrating the training workflow

