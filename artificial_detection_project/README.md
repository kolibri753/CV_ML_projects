# Task 1 - DL (MANDATORY)

This project focuses on detecting visual artifacts in AI-generated images using multiple deep learning models, anomaly detection techniques, and region-of-interest (ROI) analysis.

## Key Features
- binary classification: Clean (1) vs. Artifact (0);
- multiple model architectures: EfficientNet, ResNet, ConvNeXt, Autoencoder, CLIP;
- ensemble and selective ensemble predictions;
- GradCAM visualizations;
- ROI-based face detection preprocessing using MediaPipe.

## Final Metrics

### On Test Set (Using Validation Split)
| Model              | Accuracy | Precision | Recall | F1 Score | AUC     |
|-------------------|----------|-----------|--------|----------|---------|
| Ensemble Selective| 0.965    | 0.9626    | 1.0000 | 0.9809   | 0.8250  |
| ResNet            | 0.960    | 0.9674    | 0.9889 | 0.9780   | 0.8444  |
| Ensemble          | 0.955    | 0.9524    | 1.0000 | 0.9756   | 0.7750  |
| ConvNeXt          | 0.955    | 0.9622    | 0.9889 | 0.9753   | 0.8194  |
| EfficientNet      | 0.950    | 0.9570    | 0.9889 | 0.9727   | 0.7944  |
| CLIP              | 0.900    | 0.9000    | 1.0000 | 0.9474   | 0.5000  |
| Autoencoder       | 0.440    | 0.8400    | 0.4667 | 0.6000   | 0.3333  |

### On Full Dataset with ROI
| Model       | Accuracy | Precision | Recall | F1 Score | AUC     |
|-------------|----------|-----------|--------|----------|---------|
| EfficientNet| 0.990    | 0.9944    | 0.9944 | 0.9944   | 0.9722  |

This performance improvement may be due to the ROI-based preprocessing in the current setup or the fact that training was done directly on the full training set without splitting into a separate validation set. Ideally, the factors should have been tested separately, but due to the approaching deadline, priority was given to getting a working solution.

## How to Use (Google Colab)

You can start the Colab notebook from different sections:
- from the top (to run everything from scratch);
- from "Predictions on Validation Set" (if models are trained);
- from "ROI" section (for best performance using face-based preprocessing).

## Directory Structure
- `models/`: saved model weights (EfficientNet, ResNet, ConvNeXt, etc.);
- `gradcam_outputs/`: GradCAM visualizations from full-frame images;
- `gradcam_outputs_roi/`: GradCAM visualizations from face-cropped ROI images;
- `train.csv`, `val.csv`, `test.csv`: image-label splits;
- `trainee_dataset`: input files.

## Setup (requirements.txt)
```
torch
numpy==1.26.4
opencv-python
pandas
matplotlib
seaborn
timm
albumentations
scikit-learn
torchvision
torchcam
mediapipe
Pillow
ftfy
regex
tqdm
git+https://github.com/openai/CLIP.git
```

---

## Notes
- the dataset is highly imbalanced (1:9 ratio), tried using Focal loss and oversampling;
- CLIP and Autoencoder models performed poorly (0) compared to CNN-based classifiers;
- ROI preprocessing with EfficientNet - the best results;
- GradCAM was used for understand this all better.



