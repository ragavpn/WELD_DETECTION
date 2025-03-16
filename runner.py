import torch
import torchvision.transforms as transforms
from torchvision.models import densenet169
from PIL import Image
import torch.nn as nn
import warnings
import sys

warnings.filterwarnings("ignore")

# Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoint.pth"  # Path to the trained model checkpoint
CLASS_NAMES = [                     # Statically declaring the classes
    'Aligned Pores', 'Burn Through', 'Exces Penetration', 'Exces Reinforcement',
    'Gas Hole', 'Incomplete Penetration', 'Irregular Root', 'Lack of Fusion',
    'Multiple pores', 'Pin Holes', 'Pittings', 'Porasity', 'Under Cut',
    'Wire Stub', 'Worm Hole'
] 

# Load Model
def load_model(model_path):
    """Loads the trained model from a checkpoint."""
    model = densenet169(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(CLASS_NAMES))  # 15 classes
    model = model.to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# Image Preprocessing
def preprocess_image(image_path):
    """Applies transformations to an image for model inference."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

# Run Model on Image
def predict(image_path, model):
    """Predicts the class of the given image using the trained model."""
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return CLASS_NAMES[predicted_class.item()]

# Command-Line Execution
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python runner.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load model
    print("Loading model...")
    model = load_model(CHECKPOINT_PATH)

    # Predict class
    print(f"Predicting class for: {image_path}")
    predicted_class = predict(image_path, model)
    
    print(f"Predicted Class: {predicted_class}")
