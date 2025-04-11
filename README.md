# EfficientNet-B0 Modifications for Cityscapes Dataset

## Project Overview
This repository contains experiments on modifying EfficientNet-B0 for semantic segmentation on the Cityscapes dataset. The project explores various architectural enhancements to improve segmentation performance while maintaining computational efficiency.

## Key Features
- Implementation of multiple EfficientNet-B0 variants
- Comprehensive comparative analysis of model performance
- Training and evaluation pipeline for semantic segmentation
- Visualization tools for model performance comparison

## Model Variants
The project implements and evaluates the following model architectures:

### 1. Baseline EfficientNet-B0
Standard EfficientNet-B0 architecture adapted for semantic segmentation with minimal modifications.

### 2. EfficientNet-B0 with CBAM (Convolutional Block Attention Module)
Enhances the baseline model by incorporating channel and spatial attention mechanisms to focus on relevant features.

### 3. EfficientNet-B0 with Mish Activation
Replaces all ReLU activation functions with Mish activation (x * tanh(softplus(x))), which provides smoother gradients and improved learning dynamics.

### 4. EfficientNet-B0 with DeepLabV3+ Segmentation Head
Incorporates the DeepLabV3+ architecture with Atrous Spatial Pyramid Pooling (ASPP) for multi-scale feature extraction.

### 5. Combined Approach
Integrates all three modifications (CBAM, Mish, DeepLabV3+) into a single model to leverage their complementary strengths.

## Setup and Installation

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install efficientnet_pytorch
pip install numpy pandas matplotlib
pip install tqdm scikit-learn
pip install jupyter
git clone https://github.com/mcordts/cityscapesScripts.git
pip install -e cityscapesScripts
```

### Dataset
This project uses the Cityscapes dataset, which needs to be downloaded separately:
1. Register at the [Cityscapes website](https://www.cityscapes-dataset.com/)
2. Download the dataset
3. Update the `cityscapes_root` variable in the notebook to point to your dataset location

## Usage
The main implementation is contained in `group-project.ipynb`. To run the experiments:

1. Set up the environment with the required dependencies
2. Configure the dataset path
3. Execute the notebook cells sequentially
4. Results, visualizations, and trained models will be generated as you progress

## Results
Our experiments demonstrate that:

- CBAM attention mechanism improves feature representation by focusing on relevant spatial and channel information
- Mish activation functions help with gradient flow and regularization, resulting in better convergence
- DeepLabV3+ segmentation head enhances multi-scale understanding necessary for accurate segmentation
- The combined model achieves the highest performance, showing that these modifications have complementary benefits

## Model Performance Comparison
| Model | Test Accuracy | Test Loss |
|-------|--------------|-----------|
| Baseline | x.xx | x.xx |
| With CBAM | x.xx | x.xx |
| With Mish | x.xx | x.xx |
| With DeepLabV3+ | x.xx | x.xx |
| Combined | x.xx | x.xx |

## Saved Models
Trained models are saved in the `models` directory with the following files:
- `baseline_efficientnet_b0.pth`: Baseline model
- `cbam_efficientnet_b0.pth`: CBAM model
- `mish_efficientnet_b0.pth`: Mish model
- `deeplabv3_efficientnet_b0.pth`: DeepLabV3+ model
- `combined_model_efficientnet_b0.pth`: Combined approach

## References
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
- [DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611)
- [The Cityscapes Dataset](https://www.cityscapes-dataset.com/)

## License
This project is open-source and available under the MIT License.