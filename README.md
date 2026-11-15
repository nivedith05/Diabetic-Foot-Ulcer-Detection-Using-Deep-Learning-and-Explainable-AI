# Diabetic Foot Ulcer Detection Using Deep Learning & Explainable AI

<p align="center">
  <strong>A Comparative Deep Learning Approach for Automated Diabetic Foot Ulcer Detection</strong>
</p>

## Overview

Diabetic Foot Ulcers (DFUs) are a serious complication of diabetes that can lead to infections and other severe health consequences if not identified and managed appropriately.

This project explores a deep learning-based approach for automated diabetic foot ulcer image classification using Convolutional Neural Networks (CNNs) and transfer learning.

Three pretrained CNN architectures are implemented and comparatively evaluated:

- VGG16
- MobileNetV2
- EfficientNetB3

The project also incorporates Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize image regions contributing to model predictions, improving the interpretability of the classification process.

## Objectives

- Develop an automated image-based DFU classification system.
- Implement and compare three transfer-learning architectures.
- Evaluate model performance using standard classification metrics.
- Generate training and validation accuracy/loss curves.
- Analyze classification performance using confusion matrices.
- Apply Grad-CAM for visual explainability.
- Test trained models using external image inputs.

## System Architecture

The system follows a modular pipeline:

1. Input image acquisition
2. Image preprocessing and augmentation
3. CNN-based feature extraction
4. Binary classification
5. Model evaluation
6. Grad-CAM visualization
7. Prediction output

## Models Implemented

| Architecture | Description |
|---|---|
| VGG16 | Deep CNN architecture for hierarchical visual feature extraction |
| MobileNetV2 | Lightweight CNN architecture using inverted residual blocks |
| EfficientNetB3 | Efficient CNN architecture using compound scaling |

## Explainable AI

Grad-CAM is integrated to generate visual heatmaps highlighting image regions that contribute to the model's predictions.

This provides a qualitative interpretation of model behaviour rather than treating predictions as completely opaque outputs.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training and Validation Loss

## Technology Stack

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV

## Repository Structure

```text
configs/          Configuration files
dataset/          Dataset instructions
models/           Saved model instructions
notebooks/        Experiment notebooks
src/              Source code
  preprocessing/  Image preprocessing
  training/       CNN training scripts
  evaluation/     Model evaluation
  explainability/ Grad-CAM implementation
  prediction/     Image prediction
evaluation/       Training histories and results
outputs/          Prediction and Grad-CAM outputs
docs/             Research documentation
assets/           Diagrams and screenshots
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/dfu-detection-explainable-ai.git
cd dfu-detection-explainable-ai
```

Create a virtual environment:

```bash
python -m venv env
```

Activate the environment on Windows:

```bash
env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Organize the dataset using the following structure:

```text
dataset/
├── train/
│   ├── abnormal/
│   └── normal/
├── validation/
│   ├── abnormal/
│   └── normal/
└── test/
    ├── abnormal/
    └── normal/
```

Dataset files are not included in this repository. Refer to `dataset/README.md` for dataset source, access instructions and licensing information.

## Model Training

Run the respective training scripts:

```bash
python src/training/train_vgg16.py
python src/training/train_mobilenetv2.py
python src/training/train_efficientnetb3.py
```

## Prediction

Run the prediction module:

```bash
python src/prediction/predict.py
```

Follow the script's instructions to provide an input image.

## Grad-CAM Visualization

Run the explainability module:

```bash
python src/explainability/gradcam.py
```

Generated visualizations can be stored in the `outputs/gradcam/` directory.

## Results

Model comparison results will be reported using the same independent test dataset and evaluation protocol.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| VGG16 | To be added | To be added | To be added | To be added |
| MobileNetV2 | To be added | To be added | To be added | To be added |
| EfficientNetB3 | To be added | To be added | To be added | To be added |

*Replace the placeholders with verified final test-set metrics, not training or validation accuracy.*

## Research Documentation

The project is supported by an IEEE-style research paper documenting the methodology, system architecture, comparative evaluation and explainability analysis.

Refer to `docs/research_paper/` for the research document when available.

## Limitations

- Predictions depend on the quality and diversity of the training dataset.
- Performance on external images may differ from validation results.
- Grad-CAM visualizations provide qualitative explanations and do not establish clinical reliability.
- This project is intended for academic research and educational purposes, not independent medical diagnosis.

## Future Improvements

- Integration of a web-based prediction interface.
- Evaluation using larger and more diverse datasets.
- Further model optimization.
- Improved explainability and localization techniques.
- External validation across different image sources.

## License

This project is intended for academic and research purposes. Refer to the repository license and dataset licensing terms before redistribution.

## Acknowledgements

We acknowledge the researchers and open-source communities whose contributions to deep learning, computer vision and explainable AI supported this project.

---

**Developed as an academic research project in Deep Learning, Computer Vision and Explainable Artificial Intelligence.**
