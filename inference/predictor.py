import torch
from PIL import Image
from models.loader import load_model
from utils.image_preprocessing import preprocess_image
from utils.class_map import class_map

def predict(image: Image.Image, model_name: str):
    model = load_model(model_name)
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_idx = probs.argmax().item()
        confidence = probs[predicted_class_idx].item()

        label = class_map.get(predicted_class_idx, "Unknown")

    return {
        "predicted_class": predicted_class_idx,
        "label": label,
        "confidence": round(confidence, 4)
    }
