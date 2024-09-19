# AutoPET III Challenge: Final Submission

This repository contains the code and models used for the final submission in the **AutoPET III Challenge**. This repository is released under the MIT License.

## Overview

Our method uses a classifier to differentiate between FDG and PSMA tracers. It then runs inference on the PET/CT using a tracer-specific nnU-Net ensemble.
The paper is available at: https://arxiv.org/pdf/2409.12155

### Segmentation Models:
- **FDG Model**: nnUNet ensemble specifically trained with FDG PET data.
- **PSMA Models**: Includes two models trained on PSMA PET data:
  1. A standard nnU-Net architecture.
  2. A nnU-Net model with a Residual Encoder architecture.

### Classifier:
- **Tracer Classifier**: A model trained to classify the input as either FDG or PSMA tracer. This classifier can be used if the used tracer is unknown.

## Model Checkpoints

All model weights are available under https://drive.google.com/file/d/1nY7ciiJPcfxtv1XFpY-eWsmBkxfTSJez/view.
They include the following files/folders:

- **FDG Model**: Checkpoints are found in the `Dataset001_fdgweighted` folder.
- **PSMA Models**:
  - Standard nnU-Net: Located in `Dataset002_psmaweighted`.
  - Residual Encoder nnU-Net: Located in `Dataset003_psmaweighted`.
- **Tracer Classifier**: Model weights for the tracer classifier are available in `tracer_classifier.pt`.

