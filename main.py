from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import firebase_admin
from firebase_admin import credentials, initialize_app
from inference.predictor import predict
from dotenv import load_dotenv
import os, json, base64, uvicorn

load_dotenv()

# Firebase Admin Initialization
if not len(firebase_admin._apps):
    b64_creds = os.environ["FIREBASE_ADMIN_CREDENTIALS_B64"]
    json_creds = json.loads(base64.b64decode(b64_creds).decode())
    cred = credentials.Certificate(json_creds)
    initialize_app(cred)

app = FastAPI(title="Crop Disease Detection API")

# ✅ CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with actual frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

