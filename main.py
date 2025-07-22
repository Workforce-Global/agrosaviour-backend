from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import firebase_admin
from inference.predictor import predict
import os
import uvicorn
import os, json, base64
from firebase_admin import credentials, initialize_app
from dotenv import load_dotenv
load_dotenv()

# Only initialize once
if not len(firebase_admin._apps):
    b64_creds = os.environ["FIREBASE_ADMIN_CREDENTIALS_B64"]
    json_creds = json.loads(base64.b64decode(b64_creds).decode())
    cred = credentials.Certificate(json_creds)
    initialize_app(cred)


app = FastAPI(title="Crop Disease Detection API")

@app.post("/predict/")
async def predict_crop_disease(
    file: UploadFile = File(...),
    model_name: str = "efficientnet"  # Options: mobilenet, efficientnet
):
    if model_name not in ["mobilenet", "efficientnet", "MobileNet", "EfficientNet"]:
        raise HTTPException(status_code=400, detail="Invalid model name")

    try:
        image = Image.open(file.file).convert("RGB")
        result = predict(image, model_name)
        return {
            "model_used": model_name,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Updated Section for GCP usage

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

