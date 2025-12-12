import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

IMG_SIZE = (224, 224)

def confidence_metric(y_true, y_pred):
    max_prob = tf.reduce_max(y_pred, axis=-1)
    return tf.reduce_mean(max_prob)

# تحميل أسماء الفئات
with open("labels.txt", "r", encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

app = FastAPI()

# (اختياري) CORS عشان Flutter يقدر يرسل طلبات بدون مشاكل
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # لاحقًا خليها دومين تطبيقك فقط
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل النموذج مرة واحدة عند تشغيل السيرفر
MODEL_PATH = "SavedModel"
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"confidence_metric": confidence_metric},
)

def prepare_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.array(img).astype("float32")  # 0..255
    x = preprocess_input(x)              # -> [-1, 1]
    x = np.expand_dims(x, 0)             # [1,224,224,3]
    return x

@app.get("/")
def root():
    return {"message": "API is running. Use /status or /health or POST /predict"}

@app.get("/status")
def status():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok", "classes": len(CLASS_NAMES)}

@app.get("/plant")
def plant():
    return {"message": "plant endpoint works (GET)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 5):
    try:
        image_bytes = await file.read()
        x = prepare_image(image_bytes)

        probs = model.predict(x, verbose=0)[0].astype(float)

        top_idx = int(np.argmax(probs))
        top = {"label": CLASS_NAMES[top_idx], "confidence": float(probs[top_idx])}

        k = max(1, min(int(topk), len(CLASS_NAMES)))
        idxs = np.argsort(probs)[::-1][:k]
        results = [{"label": CLASS_NAMES[i], "confidence": float(probs[i])} for i in idxs]

        return JSONResponse({"top": top, "results": results})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# للتشغيل المحلي فقط (ليس Render)
if name == "main":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)