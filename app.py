from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "mango_model.pt"
model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
model.eval()

classes = [
    "Anthracnose fruit",
    "Anthracnose leaf",
    "Bacterial Canker fruit",
    "Gall_Mid leaf",
    "Healthy fruit",
    "Healthy leaf"
]

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

class AdviceRequest(BaseModel):
    disease: str
    confidence: float
    health_score: float

@app.get("/")
def home():
    return {"message": "Mango Disease API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400,
                            detail="Only JPEG, PNG, or WEBP images are supported")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf_tensor, pred = torch.max(probs, 1)
        label = classes[pred.item()]
        confidence = round(float(conf_tensor.item()) * 100, 2)
        if "Healthy" in label:
            health_score = round(90 + (confidence / 100) * 10, 2)
        else:
            health_score = round(100 - confidence, 2)
        return {"disease": label, "confidence": confidence, "health_score": health_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/advice")
async def get_advice(req: AdviceRequest):
    prompt = f"""You are an agricultural expert. Answer about the mango disease below.
Disease: {req.disease}
Confidence: {req.confidence}%
Health Score: {req.health_score}/100
Use EXACTLY these 5 sections:
What is it?
Possible Causes
Treatment
Prevention
Is it safe to eat?"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openrouter/auto",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.7
                }
            )
        if response.status_code != 200:
            raise HTTPException(status_code=500,
                                detail=f"AI error: {response.status_code}")
        data = response.json()
        advice = data["choices"][0]["message"]["content"]
        return {"advice": advice}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
