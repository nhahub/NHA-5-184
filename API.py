
from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
import cv2

from PREPOCESSING import preprocess_rgb

app = FastAPI(
    title="RetinoGrade AI API",
    description="Diabetic Retinopathy Grading API",
    version="1.0"
)

# DR class labels
CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

# Load the existing trained model (unchanged)
model = tf.keras.models.load_model(
    "models/B_full_s42_384_mix_model.keras",
    compile=False
)


@app.get("/")
def home():
    return {
        "message": "RetinoGrade AI API is running!"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    # Decode image
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

    # Preprocess image (384x384)
    processed = preprocess_rgb(
        img,
        target_size=384
    )

    # Convert pixel range from 0-1 to 0-255
    processed = processed * 255.0

    # Add batch dimension
    processed = np.expand_dims(
        processed,
        axis=0
    )

    # Model prediction
    predictions = model.predict(
        processed,
        verbose=0
    )[0]

    # Get predicted class
    predicted_class = int(np.argmax(predictions))

    # Get confidence
    confidence = float(predictions[predicted_class])

    # Return results
    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "predicted_grade": CLASS_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(prob) * 100, 2)
            for i, prob in enumerate(predictions)
        }
    }
