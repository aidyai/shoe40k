
from PIL import Image
import requests

import torch
from model import ColaModel
from data import DataModule
import sys



import torch
import torchvision.transforms as transforms
from PIL import Image

class ImageClassifier:
    def __init__(self, model_path, class_names):
        self.model_path = model_path
        self.class_names = class_names

        # Load the model
        self.model = torch.load(model_path)
        self.model.eval()
        self.softmax = torch.nn.Softmax(dim=0)

        # Define preprocessing steps for input images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path):
        # Load and preprocess the input image
        image = Image.open(image_path)
        image = self.transform(image)
        image = image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(image)

        # Apply softmax to get probabilities
        probabilities = self.softmax(output[0])

        # Get predicted class index
        _, predicted_idx = torch.max(probabilities, 0)

        # Prepare predictions
        predictions = []
        for idx, prob in enumerate(probabilities):
            predictions.append({
                "class": self.class_names[idx],
                "probability": prob.item()
            })

        return predictions

# Example usage:
class_names = ["Boot", "Dressing Shoe", "Heels", "Sandals", "Sneakers"]
model_path = './.ckpt'  # Update with your model path
image_path = './.jpg'  # Update with your image path

classifier = ImageClassifier(model_path, class_names)
predictions = classifier.predict(image_path)
print(predictions)
