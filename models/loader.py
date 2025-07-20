import torch
from torchvision import models

def load_model(model_name: str):
    if model_name == "mobilenet" or model_name == "MobileNet":
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 18)
        model.load_state_dict(torch.load("models/mobilenetv3_crop_disease.pth", map_location="cpu"))
    elif model_name == "efficientnet" or model_name == "EfficientNet":
        model = models.efficientnet_b4(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 18)
        model.load_state_dict(torch.load("models/best_efficientnetb4.pth", map_location="cpu"))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.eval()
    return model
