
import json
import base64
from pathlib import Path

import cv2
import keras
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from PREPOCESSING import preprocess_rgb
from explain.gradcam import gradcam, overlay


app = FastAPI(
    title="RetinoGrade AI API",
    description="Diabetic Retinopathy Grading API",
    version="2.0"
)


# -----------------------------
# Configuration
# -----------------------------

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

QUALITY_CLASS_NAMES = [
    "Good",
    "Usable",
    "Reject"
]

QUALITY_MESSAGES = {
    "Good": "Image quality is good.",
    "Usable": "Image quality is limited. The result needs a human review.",
    "Reject": "Image quality is too low to grade. Please take a new photo."
}

MIN_SIDE = 64

BASE_DIR = Path(__file__).resolve().parent


# -----------------------------
# Load DR model
# -----------------------------

model = tf.keras.models.load_model(
    BASE_DIR / "models" / "B_full_s42_384_mix_model.keras",
    compile=False
)


# -----------------------------
# Load Quality model
# -----------------------------

quality_path = BASE_DIR / "models" / "quality_b0_224.keras"

with open(quality_path / "config.json", "r") as f:
    quality_config = json.load(f)

quality_model = keras.saving.deserialize_keras_object(
    quality_config
)

quality_model.load_weights(
    quality_path / "model.weights.h5"
)


# -----------------------------
# Home
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "RetinoGrade AI API is running!"
    }


# -----------------------------
# Prediction
# -----------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    img = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # Convert BGR to RGB
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    height, width = img.shape[:2]

    # -----------------------------
    # Minimum image size check
    # -----------------------------

    if min(height, width) < MIN_SIDE:
        return {
            "filename": file.filename,
            "quality": {
                "class": "Reject",
                "message": "Image is too small. Please take a new photo."
            },
            "dr": None,
            "gradcam": None
        }

    # -----------------------------
    # Quality prediction
    # -----------------------------

    quality_img = cv2.resize(
        img,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    quality_input = np.expand_dims(
        quality_img.astype(np.float32),
        axis=0
    )

    quality_predictions = quality_model.predict(
        quality_input,
        verbose=0
    )[0]

    quality_class = int(
        np.argmax(quality_predictions)
    )

    quality_label = QUALITY_CLASS_NAMES[quality_class]

    quality_confidence = float(
        quality_predictions[quality_class]
    )

    quality_result = {
        "class": quality_label,
        "confidence": round(quality_confidence * 100, 2),
        "message": QUALITY_MESSAGES[quality_label],
        "probabilities": {
            QUALITY_CLASS_NAMES[i]: round(
                float(prob) * 100, 2
            )
            for i, prob in enumerate(quality_predictions)
        }
    }

    # -----------------------------
    # Reject poor-quality images
    # -----------------------------

    if quality_label == "Reject":
        return {
            "filename": file.filename,
            "quality": quality_result,
            "dr": None,
            "gradcam": None
        }

    # -----------------------------
    # DR preprocessing
    # -----------------------------

    processed = preprocess_rgb(
        img,
        target_size=384
    )

    processed = processed * 255.0

    # -----------------------------
    # DR prediction
    # -----------------------------

    predictions = model.predict(
        np.expand_dims(processed, axis=0),
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_class]
    )

    dr_result = {
        "predicted_class": predicted_class,
        "predicted_grade": CLASS_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(
                float(prob) * 100, 2
            )
            for i, prob in enumerate(predictions)
        }
    }

    # -----------------------------
    # Grad-CAM generation
    # -----------------------------

    heatmap, gradcam_probs, target = gradcam(
        model,
        processed,
        target=predicted_class
    )

    gradcam_image = overlay(
        processed,
        heatmap
    )

    # Convert RGB overlay to PNG
    success, encoded_image = cv2.imencode(
        ".png",
        cv2.cvtColor(
            gradcam_image,
            cv2.COLOR_RGB2BGR
        )
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Could not generate Grad-CAM image."
        )

    gradcam_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")

    gradcam_result = {
        "target_class": int(target),
        "target_grade": CLASS_NAMES[int(target)],
        "image": "data:image/png;base64," + gradcam_base64
    }

    # -----------------------------
    # Final response
    # -----------------------------

    return {
        "filename": file.filename,
        "quality": quality_result,
        "dr": dr_result,
        "gradcam": gradcam_result
    }
