# Weld Defect Detection Using Deep Learning

## Project Overview
This project introduces a sophisticated deep learning solution for automated weld defect detection and classification using X-ray imagery. The system employs a two-stage approach: first identifying the presence of any defect, followed by precise classification of the specific defect type. This methodology ensures high accuracy while maintaining practical applicability in industrial settings.

## Technical Architecture

### Base Model
At the core of our system lies a DenseNet169 architecture, pretrained on ImageNet and fine-tuned for our specific use case. We chose DenseNet169 for its exceptional feature extraction capabilities and proven performance in medical and industrial imaging tasks. The model's dense connectivity pattern allows for better feature reuse and gradient flow, making it particularly suitable for detecting subtle defect patterns in X-ray images.

### Dual-Head Classification
The model implements a novel dual-head architecture:
```python
class TwoStageWeldModel(nn.Module):
    def __init__(self, num_classes, num_anomaly_classes):
        super(TwoStageWeldModel, self).__init__()
        self.base_model = densenet169(pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_size = self.base_model.classifier.in_features
        self.anomaly_classifier = nn.Linear(self.feature_size, 2)
        self.specific_classifier = nn.Linear(self.feature_size, num_classes)
```

The first head performs binary classification (defect/no-defect), while the second head specializes in identifying specific defect types. This approach allows the model to first focus on detecting anomalies before attempting detailed classification, mirroring human expert methodology.

## Data Processing and Augmentation

### Dataset Organization
The training data is organized into class-specific directories, with the following defect categories:
- Burn Through
- Excess Penetration
- Gas Hole
- Lack of Fusion
- Multiple Pores
- No Anomaly (Normal welds)

### Intelligent Data Handling
Our custom dataset class implements sophisticated data handling:
```python
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
])
```

To address class imbalance, we implement dynamic augmentation that automatically balances class distributions while preserving image characteristics critical for defect detection.

## Training Methodology

### Loss Function Design
The training process utilizes a composite loss function that balances both classification tasks:
- Binary Cross-Entropy for anomaly detection
- Cross-Entropy for specific defect classification

### Optimization Strategy
The model is trained using the Adam optimizer with carefully tuned hyperparameters:
- Learning Rate: 0.0001
- Batch Size: 32
- Training Duration: 100 epochs

### Checkpoint Management
We implement a robust checkpoint system that:
- Saves regular progress during training
- Maintains the best-performing model state
- Enables training resumption from interruptions
- Tracks performance metrics across training sessions

## Performance Metrics

Our latest model evaluation demonstrates exceptional performance:
- Binary Anomaly Detection: 99.6% accuracy
- Specific Defect Classification: 83.2% accuracy
- Overall Two-Stage Classification: 83.2% accuracy

These metrics indicate strong performance in both general anomaly detection and specific defect classification, making the system suitable for industrial deployment.

## Practical Implementation

### Hardware Requirements
- CUDA-compatible GPU
- Minimum 6GB GPU memory
- 16GB system RAM recommended

### Software Dependencies
- PyTorch (1.7+)
- torchvision
- scikit-learn
- PIL
- tqdm
- matplotlib
- seaborn
- numpy

### Usage Guide
1. Data Preparation:
   - Organize X-ray images into class-specific folders
   - Ensure consistent image format and size
   
2. Model Training:
   ```bash
   python runner.ipynb
   ```
   
3. Monitoring:
   - Real-time loss tracking
   - Automatic checkpoint saving
   - Performance visualization

4. Inference:
   - Load best_model.pth for production use
   - Batch processing available for multiple images

## Future Development

We are actively working on several enhancements:
- Integration of cross-validation for improved reliability
- Expansion of data augmentation techniques
- Implementation of hyperparameter optimization
- Development of ensemble methods
- Creation of a production-ready API interface

## Deployment Considerations
The system is designed for industrial deployment with considerations for:
- Real-time processing capabilities
- Robust error handling
- Checkpoint recovery
- Scalable architecture

This solution provides a reliable, automated approach to weld defect detection, significantly reducing the dependency on manual inspection while maintaining high accuracy standards.
