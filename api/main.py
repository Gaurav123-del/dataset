from fastapi import FastAPI, File, UploadFile
from pathlib import Path
from PIL import Image
import hashlib
import json
import io
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch import nn

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "intel_efficientnet_b0_v1_best.pth"
MODEL_NAME = "EfficientNet-B0 V1"
CACHE_FILE = BASE_DIR / "cache" / "prediction_cache.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224

classes = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CACHE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

if not CACHE_FILE.exists():
    with open(CACHE_FILE, "w") as f:
        json.dump({}, f)


def load_cache():
    with open(CACHE_FILE, "r") as f:
        return json.load(f)


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


def generate_fingerprint(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


app = FastAPI(
    title="Intel Image Classification API",
    description="EfficientNet-B0 Image Classification with Fingerprinting and Caching",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "Intel Image Classification API",
        "model": MODEL_NAME,
        "status": "running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    fingerprint = generate_fingerprint(image_bytes)

    cache = load_cache()

    if fingerprint in cache:

        cached_result = cache[fingerprint]

        return {
            "filename": file.filename,
            "fingerprint": fingerprint,
            "prediction": cached_result["prediction"],
            "confidence": cached_result["confidence"],
            "model": cached_result["model"],
            "cached": True
        }

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_tensor = test_transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = classes[
        predicted_index.item()
    ]

    confidence_value = confidence.item()

    cache[fingerprint] = {
        "prediction": predicted_class,
        "confidence": confidence_value,
        "model": MODEL_NAME
    }

    save_cache(cache)

    return {
        "filename": file.filename,
        "fingerprint": fingerprint,
        "prediction": predicted_class,
        "confidence": confidence_value,
        "model": MODEL_NAME,
        "cached": False
    }