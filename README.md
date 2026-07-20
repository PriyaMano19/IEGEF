# Iterative Explainability-Guided Enhancement Framework (IEGEF)

## Overview

This repository contains the implementation of the **Iterative Explainability-Guided Enhancement Framework (IEGEF)** developed as part of a Postgraduate Diploma research project. The framework aims to improve the interpretability of deep learning models for chest X-ray classification by integrating explainability feedback into the training process.

A pretrained **ResNet50** model is used as the baseline classifier for detecting four chest X-ray categories:
- COVID-19
- Normal
- Lung Opacity
- Viral Pneumonia

The proposed IEGEF framework extracts Grad-CAM attention maps during training and optimizes them using lung segmentation masks through an attention-guided Dice loss. This iterative optimization encourages the model to focus on clinically relevant lung regions while maintaining classification performance.

## Features

- Baseline ResNet50 image classification
- Grad-CAM, Grad-CAM++, and Integrated Gradients explainability
- Lung mask-guided attention optimization
- Iterative Explainability-Guided Enhancement Framework (IEGEF)
- Explainability evaluation using:
  - Intersection over Union (IoU)
  - Pointing Game
  - Heatmap Coverage
- Statistical comparison between baseline and IEGEF
- Qualitative visualization of attention maps
- Automated experiment notebooks and evaluation pipeline

## Dataset

- COVID-19 Radiography Dataset
- Four-class chest X-ray classification
- Lung segmentation masks used for attention-guided optimization

## Project Structure

```
notebooks/
utils/
datasets/
checkpoints/
results/
```

## Experimental Pipeline

1. Data preprocessing
2. Baseline ResNet50 training
3. Explainability evaluation
4. IEGEF optimization
5. Retraining
6. Quantitative evaluation
7. Statistical analysis
8. Qualitative analysis

## Results

The proposed IEGEF framework improves the localization quality of Grad-CAM-based explanations while maintaining competitive classification performance. Experimental results demonstrate higher IoU and improved alignment of attention maps with annotated lung regions compared to the baseline model.

## Technologies

- Python
- PyTorch
- Torchvision
- Captum
- Grad-CAM
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Dataset
Download the COVID-19 Radiography Database from Kaggle:

https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Extract it into:

datasets/COVID-19_Radiography_Dataset/