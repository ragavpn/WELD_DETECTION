import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet169
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
from tqdm import tqdm
import warnings

# Define Hyperparameters
hyperparams = {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "checkpoint_path": "checkpoint.pth",
    "hyperparam_file": "hyperparams.txt",
    "train_data_path": "/home/ragavpn/Desktop/HYPOTHESIS/CODING/PROJECTS/BHEL/Dataset/Train",  # Update with actual dataset path
    "train_split": 0.8  # 80% training, 20% testing
}

warnings.filterwarnings("ignore")
num_epochs = 10

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.anomaly_labels = []  # Binary labels: 0 for No Anomaly, 1 for any anomaly

        # Get class names dynamically from subfolders
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Find index of "No Anomaly" class
        self.no_anomaly_idx = self.class_to_idx.get("No Anomaly", -1)
        
        # Get anomaly class names (excluding "No Anomaly")
        self.anomaly_classes = [cls for cls in self.classes if cls != "No Anomaly"]
        self.num_anomaly_classes = len(self.anomaly_classes)
        self.anomaly_class_to_idx = {cls: i for i, cls in enumerate(self.anomaly_classes)}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                self.image_paths.append(img_path)
                
                # Original class label
                class_idx = self.class_to_idx[cls]
                self.labels.append(class_idx)
                
                # Binary anomaly label (0 for No Anomaly, 1 for any anomaly)
                is_anomaly = 0 if cls == "No Anomaly" else 1
                self.anomaly_labels.append(is_anomaly)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        anomaly_label = self.anomaly_labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, anomaly_label


# Data Loading and Splitting
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomDataset(root_dir=hyperparams["train_data_path"], transform=transform)
num_classes = dataset.num_classes

# Split dataset into train and test sets
train_size = int(hyperparams["train_split"] * len(dataset))
test_size = len(dataset) - train_size
train_dataset = torch.utils.data.Subset(dataset, range(train_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False, num_workers=4)


# Two-Stage Model: Binary Anomaly Detection + Specific Anomaly Classification
class TwoStageWeldModel(nn.Module):
    def __init__(self, num_classes, num_anomaly_classes):
        super(TwoStageWeldModel, self).__init__()
        
        # Base model (feature extractor) - using DenseNet169
        self.base_model = densenet169(pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Common feature size
        self.feature_size = self.base_model.classifier.in_features
        
        # First stage: Binary anomaly classifier
        self.anomaly_classifier = nn.Linear(self.feature_size, 2)  # 0: No Anomaly, 1: Anomaly
        
        # Second stage: Specific anomaly classifier
        self.specific_classifier = nn.Linear(self.feature_size, num_classes)

    def forward(self, x):
        # Extract features
        features = self.features(x)
        features = self.avg_pool(features)
        features = features.view(features.size(0), -1)
        
        # Stage 1: Binary anomaly detection
        anomaly_logits = self.anomaly_classifier(features)
        
        # Stage 2: Specific anomaly classification
        specific_logits = self.specific_classifier(features)
        
        return anomaly_logits, specific_logits

# Model Initialization
model = TwoStageWeldModel(num_classes=num_classes, num_anomaly_classes=dataset.num_anomaly_classes)
model = model.to(device)

# Define loss functions and optimizer
anomaly_criterion = nn.CrossEntropyLoss()
specific_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])


# Hyperparameter Handling
def save_hyperparams():
    """Save hyperparameters to a text file."""
    with open(hyperparams["hyperparam_file"], "w") as f:
        for key, value in hyperparams.items():
            f.write(f"{key}={value}\n")

def load_hyperparams():
    """Load hyperparameters from a text file, if it exists."""
    if not os.path.exists(hyperparams["hyperparam_file"]):
        return None
    params = {}
    with open(hyperparams["hyperparam_file"], "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            if value.replace('.', '', 1).isdigit():  # Check if it's a number
                value = float(value) if '.' in value else int(value)
            params[key] = value
    return params

def hyperparams_match():
    """Check if current hyperparameters match saved ones."""
    saved_params = load_hyperparams()
    if saved_params is None:
        return False
    return all(hyperparams[key] == saved_params.get(key) for key in hyperparams)


# Load Checkpoint If Hyperparams Match
if os.path.exists(hyperparams["checkpoint_path"]) and hyperparams_match():
    checkpoint = torch.load(hyperparams["checkpoint_path"])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded!")
else:
    print("No matching checkpoint found. Starting fresh...")
    save_hyperparams()  # Save new hyperparameters


# Training Loop with Keyboard Interrupt Handling
try:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_anomaly_loss = 0.0
        running_specific_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, labels, anomaly_labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                anomaly_labels = anomaly_labels.to(device)

                optimizer.zero_grad()
                
                # Forward pass
                anomaly_outputs, specific_outputs = model(images)
                
                # Calculate losses
                anomaly_loss = anomaly_criterion(anomaly_outputs, anomaly_labels)
                specific_loss = specific_criterion(specific_outputs, labels)
                
                # Combined loss (can adjust weights if needed)
                loss = anomaly_loss + specific_loss
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_anomaly_loss += anomaly_loss.item()
                running_specific_loss += specific_loss.item()
                
                pbar.set_postfix(loss=loss.item(), 
                                 anomaly_loss=anomaly_loss.item(),
                                 specific_loss=specific_loss.item())
                pbar.update(1)

        avg_loss = running_loss / len(train_loader)
        avg_anomaly_loss = running_anomaly_loss / len(train_loader)
        avg_specific_loss = running_specific_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Total Loss: {avg_loss:.4f}, "
              f"Anomaly Loss: {avg_anomaly_loss:.4f}, "
              f"Specific Loss: {avg_specific_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, hyperparams["checkpoint_path"])

    print(f"Training complete. Checkpoint saved at {hyperparams['checkpoint_path']}")
    print(f"Detected {num_classes} classes: {dataset.classes}")
    print(f"Anomaly classes: {dataset.anomaly_classes}")

except KeyboardInterrupt:
    print("\nTraining interrupted. Saving checkpoint...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }
    torch.save(checkpoint, hyperparams["checkpoint_path"])
    print(f"Checkpoint saved. You can resume training later.")


# Evaluate Model on Test Set
def evaluate_model(model, test_loader):
    """Evaluate the trained model on the test set."""
    model.eval()
    total = 0
    
    # For binary anomaly detection
    anomaly_correct = 0
    
    # For specific classification
    specific_correct = 0
    
    # For two-stage evaluation
    two_stage_correct = 0
    
    with torch.no_grad():
        for images, labels, anomaly_labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)
            anomaly_labels = anomaly_labels.to(device)
            
            # Forward pass
            anomaly_outputs, specific_outputs = model(images)
            
            # Binary anomaly prediction
            _, anomaly_preds = torch.max(anomaly_outputs, 1)
            anomaly_correct += (anomaly_preds == anomaly_labels).sum().item()
            
            # Specific anomaly prediction
            _, specific_preds = torch.max(specific_outputs, 1)
            specific_correct += (specific_preds == labels).sum().item()
            
            # Two-stage evaluation:
            # If binary classifier predicts "No Anomaly", use that result
            # If binary classifier predicts "Anomaly", use the specific classifier
            two_stage_preds = specific_preds.clone()
            
            # For samples predicted as "No Anomaly" (0), set prediction to No Anomaly class
            two_stage_preds[anomaly_preds == 0] = dataset.no_anomaly_idx
            
            two_stage_correct += (two_stage_preds == labels).sum().item()
            
            total += labels.size(0)

    anomaly_accuracy = 100 * anomaly_correct / total
    specific_accuracy = 100 * specific_correct / total
    two_stage_accuracy = 100 * two_stage_correct / total
    
    return {
        "anomaly_accuracy": anomaly_accuracy,
        "specific_accuracy": specific_accuracy,
        "two_stage_accuracy": two_stage_accuracy
    }

# Compute and display accuracies
accuracy_metrics = evaluate_model(model, test_loader)
print(f"Binary Anomaly Detection Accuracy: {accuracy_metrics['anomaly_accuracy']:.2f}%")
print(f"Specific Anomaly Classification Accuracy: {accuracy_metrics['specific_accuracy']:.2f}%")
print(f"Two-Stage Classification Accuracy: {accuracy_metrics['two_stage_accuracy']:.2f}%")
