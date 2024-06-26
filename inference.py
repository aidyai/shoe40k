import requests
from src.model import Shoe40kClassificationModel
import torch
import torchvision.transforms as transforms
from PIL import Image

class Inference:
    def __init__(self, model_path, class_names):
        self.model_path = model_path
        self.class_names = class_names

        # Load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Shoe40kClassificationModel.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.softmax = torch.nn.Softmax(dim=0)

        # Define preprocessing steps for input images
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path):
        # Load and preprocess the input image
        image = Image.open(image_path)
        image = self.transform(image).to(self.device)  # Move input tensor to the same device as the model
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
class_names = ["Boot", "Dressing Shoe", "Heels", "Sandals", "Sneakers", "Crocs"]
model_path = '/content/shoe40k/artifacts/model-pgqicubi:v0/model.ckpt'  # Update with your model path
image_path = '/content/1.jpg'  # Update with your image path

classifier = Inference(model_path, class_names)
predictions = classifier.predict(image_path)
print(predictions)
