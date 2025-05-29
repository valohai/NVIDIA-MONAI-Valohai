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

1. **Preprocess Dataset**

   * Downloads the Task03\_Liver dataset from the Medical Segmentation Decathlon
   * Applies preprocessing transforms (resampling, cropping, resizing, etc.)
   * Creates manifest files for training, validation, and testing splits

2. **Train Model**

   * Trains a UNet model on the preprocessed dataset
   * Configurable number of epochs, learning rate, batch size, in/out channels

3. **Evaluate Model**

   * Evaluates model performance on the test set using Dice Score, IoU
   * Generates logs and visualizations for segmentation quality

4. **Run Inference**

   * Runs sliding window inference on new input volumes
   * Saves prediction masks and outputs visualization images

## Dataset

This project uses the **Task03\_Liver** dataset from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/). The dataset contains CT volumes of the liver with manual segmentation masks.

## Model

The training pipeline uses the **UNet** architecture implemented in [MONAI](https://monai.io/), optimized for medical image segmentation tasks.

## License

This project leverages the MONAI framework and uses datasets from the Medical Segmentation Decathlon. It is distributed under the Apache License 2.0.

## Acknowledgments

* [Project MONAI](https://github.com/Project-MONAI/MONAI)
* [Valohai](https://valohai.com/) for orchestrating the training workflow

