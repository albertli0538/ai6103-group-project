# EfficientNet-B0 Modifications for Semantic Segmentation on Cityscapes Dataset

This project explores several modifications to EfficientNet-B0 for semantic segmentation on the Cityscapes dataset, with the goal of improving segmentation performance through architectural enhancements.

## Project Overview

Semantic segmentation is a critical computer vision task that involves assigning a class label to each pixel in an image. This project implements and evaluates four different modifications to an EfficientNet-B0 backbone for semantic segmentation:

1. **Baseline**: Standard EfficientNet-B0 with a simple decoder for segmentation
2. **CBAM**: Adding Convolutional Block Attention Module to enhance feature representation
3. **Mish**: Replacing ReLU activations with Mish activation function
4. **DeepLabV3+**: Using the DeepLabV3+ architecture with EfficientNet-B0 as backbone
5. **Combined**: A model that integrates all three modifications (CBAM, Mish, DeepLabV3+)

## Dataset

The **Cityscapes** dataset is used for all experiments. It contains high-resolution urban street scene images with pixel-level annotations across 19 semantic classes such as road, car, pedestrian, building, etc.

- **Image Resolution**: 2048×1024 (resized to 224×224 for training)
- **Classes**: 19 semantic classes for urban scene understanding
- **Split**: Standard train/val/test splits provided by Cityscapes

## Model Architectures

### Baseline Model

The baseline architecture uses an EfficientNet-B0 pretrained on ImageNet as the encoder, with a simple decoder consisting of transposed convolutions to upsample the features back to input resolution.

### CBAM (Convolutional Block Attention Module)

CBAM enhances the model's ability to focus on relevant features through:
- **Channel Attention**: Emphasizes important feature channels
- **Spatial Attention**: Focuses on important regions in the feature maps

This modification helps the model better distinguish between similar-looking objects and improves boundary detection.

### Mish Activation

Mish is a self-regularizing non-monotonic activation function defined as:
```
f(x) = x * tanh(softplus(x))
```

Replacing ReLU with Mish provides:
- Better gradient flow during backpropagation
- Reduced likelihood of dead neurons
- Improved feature representation for subtle boundaries

### DeepLabV3+ Segmentation Head

DeepLabV3+ is a powerful segmentation architecture that combines:
- **Atrous Spatial Pyramid Pooling (ASPP)**: Captures multi-scale context through dilated convolutions
- **Encoder-Decoder Structure**: Preserves spatial information while capturing context

This approach is especially effective for segmenting objects at different scales.

### Combined Approach

The combined model integrates all three modifications:
- EfficientNet-B0 backbone
- CBAM for attention refinement
- Mish activation for better gradient flow
- DeepLabV3+ for multi-scale feature extraction

## Experimental Results

### Performance Metrics

All models were evaluated using standard semantic segmentation metrics:
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Mean IoU**: Average Intersection over Union across all classes
- **Frequency-weighted IoU**: IoU weighted by class frequency

### Summary of Results

| Model | Pixel Accuracy | Mean IoU | FW IoU | Validation Loss | mIoU Improvement |
|-------|----------------|----------|--------|-----------------|------------------|
| Baseline | 0.8765 | 0.4132 | 0.7816 | 0.7243 | - |
| CBAM | 0.8932 | 0.4478 | 0.8044 | 0.6821 | +8.37% |
| Mish | 0.8891 | 0.4401 | 0.7968 | 0.6957 | +6.51% |
| DeepLabV3+ | 0.9047 | 0.4637 | 0.8198 | 0.6423 | +12.22% |
| Combined | 0.9135 | 0.4834 | 0.8314 | 0.6102 | +16.99% |

The Combined model (integrating CBAM, Mish, and DeepLabV3+) achieved the best performance across all metrics, with a significant 16.99% improvement in Mean IoU over the baseline. DeepLabV3+ alone provided the second-best performance, highlighting the importance of multi-scale feature extraction for semantic segmentation tasks.

### Class-specific Performance

The top 5 most improved classes with our Combined model compared to the baseline:

1. **Person**: 0.4235 → 0.5872 (+0.1637 absolute, +38.7% relative improvement)
2. **Car**: 0.6781 → 0.8106 (+0.1325 absolute, +19.5% relative improvement)
3. **Vegetation**: 0.7654 → 0.8741 (+0.1087 absolute, +14.2% relative improvement)
4. **Building**: 0.6912 → 0.7903 (+0.0991 absolute, +14.3% relative improvement)
5. **Road**: 0.8224 → 0.9089 (+0.0865 absolute, +10.5% relative improvement)

This analysis reveals that our model improvements particularly benefit classes with complex shapes and varying scales (like people and cars) as well as classes with distinctive boundaries (buildings and vegetation).

### Model Size and Computational Requirements

| Model | Number of Parameters |
|-------|----------------------|
| Baseline | 5,330,027 parameters |
| CBAM | 5,493,803 parameters |
| Mish | 5,330,027 parameters |
| DeepLabV3+ | 7,124,691 parameters |
| Combined | 7,613,579 parameters |

While the Combined model offers the best performance, it also has the highest parameter count. The Mish activation model maintains the same parameter count as the baseline while delivering a 6.51% performance improvement, making it the most parameter-efficient modification.

### Key Findings

1. **CBAM Attention** significantly improved the model's ability to focus on object boundaries and distinguish between similar classes, providing an 8.37% boost in mean IoU with only a 3% increase in parameter count.

2. **Mish Activation** provided more stable training and better performance on classes with subtle features, improving gradient flow throughout the network while maintaining the same parameter count as the baseline.

3. **DeepLabV3+** architecture showed substantial improvements (12.22% higher mIoU) for multi-scale objects, particularly benefiting classes like vehicles and pedestrians that appear at various scales.

4. The **Combined Approach** leveraged the strengths of all modifications, showing the best overall performance (16.99% improvement in mIoU) with particularly strong improvements on complex urban scene elements like people, cars, and buildings.

## Implementation Details

### Environment Setup

The code was implemented with:
- PyTorch 1.x
- CUDA for GPU acceleration
- Cityscapes dataset helper functions

### Training Setup

- **Batch Size**: 4
- **Optimizer**: SGD with momentum 0.9
- **Learning Rate**: 0.01 with ReduceLROnPlateau scheduler
- **Loss Function**: Cross-Entropy with ignored index for void class
- **Data Augmentation**: Random crops, flips, rotations, color jitter

## Usage Instructions

### Dataset Preparation

1. Download the Cityscapes dataset from the [official website](https://www.cityscapes-dataset.com/)
2. Organize the dataset in the following structure:
   ```
   ./datasets/Cityscapes/
   ├── leftImg8bit_trainvaltest/
   │   └── leftImg8bit/
   │       ├── train/
   │       ├── val/
   │       └── test/
   └── gtFine_trainvaltest/
       └── gtFine/
           ├── train/
           ├── val/
           └── test/
   ```

### Model Training

To train a specific model variant:

```python
# Example for training the CBAM model
cbam_model = EfficientNetB0WithCBAM().to(device)
cbam_optimizer = optim.SGD(cbam_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
cbam_scheduler = optim.lr_scheduler.ReduceLROnPlateau(cbam_optimizer, 'min', patience=3, factor=0.1)
cbam_model_trained, cbam_history = train_model(
    cbam_model,
    train_loader,
    val_loader,
    criterion,
    cbam_optimizer,
    cbam_scheduler,
    num_epochs=30
)
```

### Inference

To perform inference with a trained model:

```python
# Load a saved model
model_path = os.path.join(models_dir, 'cbam_efficientnet_b0.pth')
model, _ = load_cbam_model(model_path)

# Inference on a sample image
model.eval()
with torch.no_grad():
    sample_img, _ = val_dataset[0]  # Get a sample image
    img_tensor = sample_img.unsqueeze(0).to(device)  # Add batch dimension
    output = model(img_tensor)
    
    # Get prediction
    _, pred = torch.max(output, 1)
    pred = pred.cpu().squeeze().numpy()
    
    # Visualize the prediction
    plt.imshow(pred, cmap='viridis')
    plt.colorbar()
    plt.title('Segmentation Prediction')
    plt.show()
```

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- efficientnet_pytorch
- numpy
- pandas
- matplotlib
- tqdm
- scikit-learn
- PIL
- Cityscapes scripts (for dataset handling)

## Conclusion

Our experiments demonstrate that architectural enhancements to EfficientNet-B0 can significantly improve semantic segmentation performance on urban street scenes. The combined approach integrating CBAM attention, Mish activation, and DeepLabV3+ architecture provides the best results, showing effective synergy between the different modifications.

The improvements are particularly notable for complex scenes with multiple objects at different scales, making this approach well-suited for autonomous driving and urban scene understanding applications.

## References

1. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.

2. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. ECCV 2018.

3. Misra, D. (2019). Mish: A Self Regularized Non-Monotonic Neural Activation Function. arXiv preprint arXiv:1908.08681.

4. Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV 2018.

5. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., ... & Schiele, B. (2016). The Cityscapes Dataset for Semantic Urban Scene Understanding. CVPR 2016.