from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load your TensorFlow model and class names
MODEL = tf.keras.models.load_model("potatoes.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Define your endpoints
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = np.array(Image.open(BytesIO(await file.read())).convert("RGB").resize((256, 256)))
    image = image / 255.0  # Normalize the image to 0-1 range
    img_array = tf.expand_dims(image, 0)  # Expand dimensions to match the input shape
    
    predictions = MODEL.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if _name_ == "_main_":
    uvicorn.run(app, host='localhost', port=8000)
