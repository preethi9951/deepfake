# import io
# import logging
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
# import numpy as np
# import tensorflow as tf

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # Allow frontend connection
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # or specify your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load the model
# MODEL_PATH = "model/deepfake_detector.h5"  # change if your file is .keras

# try:
#     model = tf.keras.models.load_model(MODEL_PATH)
#     logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
# except Exception as e:
#     logger.error(f"❌ Error loading model: {e}")
#     model = None

# # Prediction endpoint
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     if not model:
#         return {"error": "Model not loaded on the server."}

#     try:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#         image = image.resize((224, 224))
#         img_array = np.array(image) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         prediction = model.predict(img_array)
#         score = float(prediction[0][0])

#         result = "FAKE" if score > 0.5 else "REAL"
#         confidence = round(score * 100, 2) if score > 0.5 else round((1 - score) * 100, 2)

#         return {"result": result, "confidence": confidence}

#     except Exception as e:
#         logger.error(f"❌ Prediction error: {e}")
#         return {"error": f"Prediction failed: {str(e)}"}

# @app.get("/")
# def root():
#     return {"message": "Deepfake Detection Backend is running ✅"}


















# import io
# import logging
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
# import numpy as np
# import tensorflow as tf

# # Logging setup
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # Allow frontend access
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ✅ Load trained model
# MODEL_PATH = "model/deepfake_detector.h5"

# try:
#     model = tf.keras.models.load_model(MODEL_PATH)
#     logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
# except Exception as e:
#     logger.error(f"❌ Error loading model: {e}")
#     model = None


# @app.get("/")
# def root():
#     return {"message": "Deepfake Detection Backend is running ✅"}


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     """
#     Predict whether the uploaded image is REAL or FAKE
#     """
#     if not model:
#         return {"error": "Model not loaded"}

#     try:
#         # Read and preprocess image
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#         image = image.resize((224, 224))
#         img_array = np.array(image) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # Get model prediction
#         prediction = model.predict(img_array)
#         score = float(prediction[0][0])

#         # 🧠 If training labels were {fake: 0, real: 1}
#         label = "FAKE" if score < 0.5 else "REAL"
#         confidence = round(score * 100, 2) if label == "REAL" else round((1 - score) * 100, 2)

#         logger.info(f"Prediction: {label} ({confidence}%)")
#         return {"prediction": label, "confidence": confidence}

#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#   
# 
#       return {"error": f"Prediction failed: {str(e)}"}





# import io
# import os
# import logging
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
# import numpy as np
# import tensorflow as tf

# # ---------------- LOGGING ----------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ---------------- FASTAPI SETUP ----------------
# app = FastAPI(title="Deepfake Detection API", version="1.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow frontend access
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---------------- MODEL LOADING ----------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "model",  "deepfake_detector.h5")
# try:
#     model = tf.keras.models.load_model(MODEL_PATH)
#     logger.info(f"✅ Model loaded successfully from: {MODEL_PATH}")
# except Exception as e:
#     logger.error(f"❌ Error loading model: {e}")
#     model = None

# # ---------------- ROOT ENDPOINT ----------------
# @app.get("/")
# def root():
#     return {"message": "Deepfake Detection API is running 🚀"}


# # ---------------- PREDICTION ENDPOINT ----------------
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     if model is None:
#         return {"error": "Model not loaded"}

#     try:
#         # Load and preprocess image
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#         image = image.resize((224, 224))  # Match model input size
#         img_array = np.array(image) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # Run prediction
#         prediction = model.predict(img_array)

#         # 🔍 Debug info
#         print("🔍 Raw model output:", prediction)

#         # Extract score
#         score = float(prediction[0][0])

#         # Convert to label
#         label = "FAKE" if score >= 0.5 else "REAL"

#         # Confidence calculation
#         confidence = float(score * 100 if label == "FAKE" else (1 - score) * 100)

#         # ✅ Return both
#         return {
#             "result": label,
#             "confidence": round(confidence, 2),
#         }

#     except Exception as e:
#         logger.error(f"❌ Prediction failed: {e}")
#         return {"error": f"Prediction failed: {str(e)}"}


# # ---------------- RUN SERVER ----------------
# # Run with: uvicorn server:app --reload



import io
import os
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- FASTAPI SETUP ----------------
app = FastAPI(title="Deepfake Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODEL LOADING ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "deepfake_detector.h5")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None


# ---------------- ROOT ENDPOINT ----------------
@app.get("/")
def root():
    return {"message": "Deepfake Detection API is running 🚀"}


# ---------------- PREDICTION ENDPOINT ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        # Load and preprocess the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))  # Match model input size
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)

        # Extract prediction score
        score = float(prediction[0][0])

        # ✅ Fixed logic — most CNNs output higher value for REAL
        label = "REAL" if score >= 0.5 else "FAKE"
        confidence = score * 100 if label == "REAL" else (1 - score) * 100

        # Log prediction details
        logger.info(
            f"🖼️ File: {file.filename} | Raw Score: {score:.4f} | Label: {label} | Confidence: {confidence:.2f}%"
        )

        # Return response
        return {
            "filename": file.filename,
            "result": label,
            "confidence": round(confidence, 2),
        }

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return {"error": f"Prediction failed: {str(e)}"}


# ---------------- RUN SERVER ----------------
# Run with: uvicorn server:app --reload
