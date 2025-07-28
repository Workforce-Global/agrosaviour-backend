# 🌾 AgroSaviour Backend (FastAPI)

This backend API performs image classification to detect diseases in crops commonly grown in Ghana — including **cassava**, **maize**, **tomato**, and **cashew** — using pre-trained PyTorch models (EfficientNet and MobileNet).

It is designed to integrate seamlessly with a Firebase-based frontend, such as Firebase Studio, and will optionally support Firebase Authentication token verification in future versions.

---

## ✅ Features

- ✅ Upload crop images for instant disease detection  
- ✅ Supports two models: `efficientnet` (accurate) and `mobilenet` (lightweight)  
- ✅ Preprocessing pipeline to ensure clean predictions  
- 🔜 Optional Firebase authentication token support  
- 🔜 History logging and analytics  

---

## 📁 Project Structure

.
├── main.py # FastAPI app entry point
├── inference/
│ ├── predictor.py # Core inference logic
│ └── models/
│ ├── loader.py # Loads the selected PyTorch model
│ └── best_efficientnetb4.pth
│ └── best_mobilenetv3.pth
├── utils/
│ ├── image_preprocessing.py # Image resizing, conversion, normalization
│ └── class_map.py # Class ID to human-readable disease mapping
├── requirements.txt
└── README.md
`


---

## 🧠 Supported Models

| Model Name    | Description                              | Use Case           |
|---------------|------------------------------------------|--------------------|
| `efficientnet`| High-accuracy model (EfficientNetB4)     | Default; better for serious diagnostics |
| `mobilenet`   | Lightweight and fast (MobileNetV3)       | Ideal for mobile and edge deployment   |

---

## 🚀 Getting Started

### 1. Clone and Setup

```bash
git clone https://github.com/Workforce-Global/agrosaviour-backend.git
cd agrosaviour-backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run the server
```
uvicorn main:app --host=0.0.0.0 --port=8080
```

### API Usage
POST /predict/
Upload a crop image and select which model to use for prediction.

Parameters
Query param: model_name → "efficientnet" (default) or "mobilenet"

Form field: file → Image file (JPG, PNG)

**Method:** `POST`  
**Content-Type:** `multipart/form-data`

✅ Response
```
{
    "model_used": "efficientnet",
    "result": {
        "predicted_class": 3,
        "label": "Fall Armyworm",
        "confidence": 0.9962
    }
}
```
✅ Example using cURL
```
curl -X POST https://agrosaviour-backend-947103695812.europe-west1.run.app/predict/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpeg"
```

Response Fields:
model_used: The machine learning model used for prediction (e.g., EfficientNet).
predicted_class: The internal class ID assigned to the disease.
label: The human-readable name of the disease.
confidence: The confidence score for the prediction (range: 0 to 1).
