---
title: Teachable Machine Image Classifier
emoji: 🏭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.24.0
app_file: app_gradio.py
pinned: false
license: mit
---

# Teachable Machine Image Classifier

This is a multi-class image classifier that supports Logistic Regression, Random Forest, and CNN models. It's designed to work similarly to Google's Teachable Machine but with more advanced features.

## Features

- **Multi-model Support**: Logistic Regression, Random Forest, and CNN
- **Real-time Training Progress**: See training progress as it happens
- **Webcam Integration**: Capture images directly from your webcam
- **Model Evaluation**: View accuracy metrics and confusion matrices
- **Batch Image Upload**: Upload multiple images at once

## How to Use

1. Visit the Space and wait for it to load
2. Upload an image using the interface
3. Click 'Classify Image' button
4. View predictions from available models

## Model Information

- **Logistic Regression**: Traditional ML approach, good for simpler classification tasks
- **Random Forest**: Ensemble learning method, robust and handles overfitting well
- **CNN**: Deep learning convolutional neural network, most powerful for complex image recognition

## Requirements

All requirements are specified in `requirements.txt` and will be automatically installed by Hugging Face Spaces.

## License

This project is licensed under the MIT License.