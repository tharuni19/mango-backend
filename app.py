from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
import httpx

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

# ── OpenRouter API (Free — get key at openrouter.ai) ──────────────────────────
OPENROUTER_API_KEY = "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"

class AdviceRequest(BaseModel):
    disease: str
    confidence: float
    health_score: float


@app.get("/")
def home():
    return {"message": "Mango Disease API is running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400,
                            detail="Only JPEG, PNG, or WEBP images are supported")
    try:
        image      = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs           = model(img_tensor)
            probs             = torch.softmax(outputs, dim=1)
            conf_tensor, pred = torch.max(probs, 1)

        label      = classes[pred.item()]
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

STRICT RULES:
- NO emojis anywhere
- NO greetings
- NO citation numbers like [1] [2] [3] [4] anywhere in the text
- NO dashes like --- anywhere
- Use EXACTLY these 5 section titles as plain text

What is it?
[2-3 simple sentences about this disease]

Possible Causes
- [cause 1]
- [cause 2]
- [cause 3]
- [cause 4]

Treatment
- [organic remedy with dosage]
- [second organic or chemical option]
- [third chemical fungicide option]
- [when and how to apply]

Prevention
- [tip 1]
- [tip 2]
- [tip 3]
- [tip 4]

Is it safe to eat?
[Clear yes or no and explain in 2-3 sentences]"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:5500",
                    "X-Title": "Mango Disease AI"
                },
                json={
                    "model": "openrouter/free",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.7
                }
            )

        print(f"[OpenRouter] status: {response.status_code}")

        if response.status_code != 200:
            print(f"[OpenRouter Error] {response.text}")
            raise HTTPException(status_code=500,
                                detail=f"AI error: {response.status_code} - {response.text[:200]}")

        data   = response.json()
        advice = data["choices"][0]["message"]["content"]
        print("[OpenRouter] Success")
        return {"advice": advice}

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))