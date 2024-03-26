from peft import PeftConfig, PeftModel
from PIL import Image
import requests

config = PeftConfig.from_pretrained("aisuko/"+os.getenv("WANDB_NAME"))
model = AutoModelForImageClassification.from_pretrained(
    config.base_model_name_or_path,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

inference_model = PeftModel.from_pretrained(model, "aisuko/"+os.getenv("WANDB_NAME"))


url="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
image=Image.open(requests.get(url, stream=True).raw)
image

image_processor=AutoImageProcessor.from_pretrained("aisuko/"+os.getenv("WANDB_NAME"))
encoding=image_processor(image.convert("RGB"), return_tensors="pt")


with torch.no_grad():
    outputs=inference_model(**encoding)
    logits=outputs.logits

predicted_class_idx=logits.argmax(-1).item()
inference_model.config.id2label[predicted_class_idx]
