# Water Segmentation using Multispectral and Optical Data

## Overview
This project focuses on segmenting water bodies using multispectral and optical data with a deep learning model. The dataset consists of 12-band multispectral TIFF images and corresponding binary PNG labels. A U-Net model is trained for semantic segmentation, and performance is evaluated using metrics such as IoU, precision, recall, and F1-score.

## Features
- **Data Handling**: Loads 12-band TIFF images and binary PNG masks.
- **Data Preprocessing**: Normalizes images and converts labels into binary format.
- **Visualization**: Displays individual spectral bands for better understanding.
- **Model Architecture**: Implements a U-Net model with dropout layers for regularization.
- **Training**: Uses the Adam optimizer and binary cross-entropy loss function.
- **Evaluation**: Computes IoU, precision, recall, and F1-score for model performance.

## Dependencies
Ensure the following libraries are installed:
```bash
pip install numpy tensorflow rasterio matplotlib scikit-learn
```

## Directory Structure
```
project_root/
│-- images/      # TIFF images (12-band, 128x128)
│-- labels/      # PNG binary masks (128x128)
│-- script.py    # Main Python script
```

## How to Run
1. Update the `data_dir` and `labels_dir` variables with the correct paths to images and labels.
2. Execute the script:
   ```bash
   python script.py
   ```
3. The script will:
   - Load and preprocess the dataset.
   - Display a visualization of spectral bands.
   - Train a U-Net model.
   - Evaluate performance using segmentation metrics.

## Model Architecture
- **Encoder**: Consists of convolutional layers with ReLU activation and max pooling.
- **Bottleneck**: Deep convolutional layers for feature extraction.
- **Decoder**: Uses transpose convolutions for upsampling.
- **Output Layer**: A single convolutional layer with a sigmoid activation function.

## Evaluation Metrics
The model performance is assessed using:
- **Intersection over Union (IoU)**
- **Precision**
- **Recall**
- **F1-score**

## Results
After training, the model predicts segmentation masks for test images. The evaluation script computes performance metrics for assessing the quality of segmentation.



