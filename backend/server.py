from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Import utility functions
from utils import preprocess_image, generate_gradcam, get_prediction_label

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Load trained model
MODEL_PATH = ROOT_DIR / 'model' / 'deepfake_detector.h5'
model = None

def load_model():
    """Load the trained deepfake detection model"""
    global model
    try:
        if MODEL_PATH.exists():
            model = keras.models.load_model(MODEL_PATH)
            logging.info(f"✅ Model loaded successfully from {MODEL_PATH}")
        else:
            logging.warning(f"⚠️  Model not found at {MODEL_PATH}. Please run train.py first.")
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")

# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    confidence: float
    prediction_class: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    filename: Optional[str] = None

class PredictionHistory(BaseModel):
    predictions: List[PredictionResult]
    total: int

# Basic routes
@api_router.get("/")
async def root():
    return {"message": "Deepfake Detection API", "status": "online"}

@api_router.get("/model-status")
async def model_status():
    """Check if model is loaded and ready"""
    return {
        "loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "exists": MODEL_PATH.exists()
    }

@api_router.post("/predict", response_model=PredictionResult)
async def predict_deepfake(file: UploadFile = File(...)):
    """
    Predict if uploaded image is real or fake (deepfake)
    
    Args:
        file: Uploaded image file
    
    Returns:
        Prediction result with label and confidence score
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running train.py"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        processed_img = preprocess_image(image_bytes)
        
        # Make prediction
        prediction = model.predict(processed_img, verbose=0)
        confidence = float(prediction[0][0])
        
        # Get label
        label, confidence_pct, pred_class = get_prediction_label(confidence)
        
        # Create result object
        result = PredictionResult(
            label=label,
            confidence=round(confidence_pct, 2),
            prediction_class=pred_class,
            filename=file.filename
        )
        
        # Save to database
        result_dict = result.model_dump()
        result_dict['timestamp'] = result_dict['timestamp'].isoformat()
        await db.predictions.insert_one(result_dict)
        
        logging.info(f"Prediction: {label} ({confidence_pct:.2f}%) - {file.filename}")
        
        return result
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.post("/predict-with-gradcam")
async def predict_with_visualization(file: UploadFile = File(...)):
    """
    Predict deepfake and return Grad-CAM visualization
    
    Args:
        file: Uploaded image file
    
    Returns:
        Grad-CAM heatmap overlay image
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        processed_img = preprocess_image(image_bytes)
        
        # Generate Grad-CAM
        heatmap_bytes = generate_gradcam(model, processed_img)
        
        if heatmap_bytes:
            return Response(content=heatmap_bytes, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=500, detail="Grad-CAM generation failed")
    
    except Exception as e:
        logging.error(f"Grad-CAM error: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@api_router.get("/predictions", response_model=List[PredictionResult])
async def get_predictions(limit: int = 50):
    """
    Get prediction history
    
    Args:
        limit: Maximum number of predictions to return
    
    Returns:
        List of past predictions
    """
    try:
        predictions = await db.predictions.find(
            {}, 
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit).to_list(limit)
        
        # Convert ISO strings back to datetime
        for pred in predictions:
            if isinstance(pred['timestamp'], str):
                pred['timestamp'] = datetime.fromisoformat(pred['timestamp'])
        
        return predictions
    
    except Exception as e:
        logging.error(f"Error fetching predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/stats")
async def get_statistics():
    """
    Get prediction statistics
    
    Returns:
        Statistics about predictions made
    """
    try:
        total = await db.predictions.count_documents({})
        fake_count = await db.predictions.count_documents({"prediction_class": "fake"})
        real_count = await db.predictions.count_documents({"prediction_class": "real"})
        
        return {
            "total_predictions": total,
            "fake_detected": fake_count,
            "real_detected": real_count,
            "fake_percentage": round((fake_count / total * 100) if total > 0 else 0, 2)
        }
    
    except Exception as e:
        logging.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy routes
@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
