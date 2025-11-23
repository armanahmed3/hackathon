from fastapi import FastAPI, UploadFile, File, Form, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import cv2
import numpy as np
from PIL import Image
import io
import uuid
import json
import asyncio
from typing import List, Dict, Optional
import shutil
from pathlib import Path
import base64

# Import trainers
from trainers.logistic_trainer import LogisticTrainer
from trainers.rf_trainer import RFTrainer
from trainers.cnn_trainer import CNNTrainer

# Import predictor
from inference.predictor import Predictor

app = FastAPI(title="Teachable Machine Web App")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed/train", exist_ok=True)
os.makedirs("data/processed/test", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("trainers", exist_ok=True)
os.makedirs("inference", exist_ok=True)
os.makedirs("ui", exist_ok=True)

# Store active training tasks
active_tasks = {}

class ClassCreationRequest(BaseModel):
    class_name: str

class UploadImageRequest(BaseModel):
    class_name: str

class TrainingRequest(BaseModel):
    models: List[str]
    classes: List[str]
    hyperparameters: Optional[Dict] = None

class PredictionResponse(BaseModel):
    logistic_regression: str
    random_forest: str
    cnn: str
    probabilities: Dict

# Helper functions
def sanitize_class_name(name):
    """Sanitize class name to prevent directory traversal"""
    # Remove leading/trailing whitespace
    name = name.strip()
    # If name is empty after stripping, return a default name
    if not name:
        return "unnamed_class"
    # Keep alphanumeric characters, spaces, underscores, and hyphens
    sanitized = "".join(c for c in name if c.isalnum() or c in " _-")
    # If sanitized name is empty, return a default name
    if not sanitized:
        return "class_" + str(uuid.uuid4())[:8]
    return sanitized

def validate_image(file_data):
    """Validate image format and size"""
    try:
        # Check if it's a valid image
        image = Image.open(io.BytesIO(file_data))
        # Check format
        if image.format.lower() not in ['jpeg', 'jpg', 'png']:
            return False, "Invalid image format. Only JPEG and PNG are allowed."
        
        # Check size (5MB limit)
        if len(file_data) > 5 * 1024 * 1024:
            return False, "Image size exceeds 5MB limit."
            
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

@app.post("/api/classes")
async def create_class(request: ClassCreationRequest):
    """Create a new class (folder)"""
    sanitized_name = sanitize_class_name(request.class_name)
    if not sanitized_name:
        raise HTTPException(status_code=400, detail="Invalid class name")
    
    class_path = os.path.join("data", "raw", sanitized_name)
    os.makedirs(class_path, exist_ok=True)
    
    return {"status": "ok", "class_path": class_path}

@app.get("/api/classes")
async def list_classes():
    """List classes and dataset counts"""
    raw_data_path = os.path.join("data", "raw")
    if not os.path.exists(raw_data_path):
        return {"classes": []}
    
    classes = []
    for class_name in os.listdir(raw_data_path):
        class_path = os.path.join(raw_data_path, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            classes.append({"name": class_name, "count": count})
    
    return {"classes": classes}

@app.delete("/api/classes/{class_name}")
async def delete_class(class_name: str):
    """Delete a class and all its images"""
    try:
        sanitized_name = sanitize_class_name(class_name)
        if not sanitized_name:
            raise HTTPException(status_code=400, detail="Invalid class name")
        
        # Delete from raw data
        raw_class_path = os.path.join("data", "raw", sanitized_name)
        if os.path.exists(raw_class_path):
            shutil.rmtree(raw_class_path)
        
        # Delete from processed data (both train and test)
        processed_train_path = os.path.join("data", "processed", "train", sanitized_name)
        if os.path.exists(processed_train_path):
            shutil.rmtree(processed_train_path)
            
        processed_test_path = os.path.join("data", "processed", "test", sanitized_name)
        if os.path.exists(processed_test_path):
            shutil.rmtree(processed_test_path)
        
        return {"status": "ok", "message": f"Class '{sanitized_name}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting class: {str(e)}")

@app.post("/api/upload")
async def upload_image(class_name: str = Form(...), file: UploadFile = File(...)):
    """Upload image to a class"""
    sanitized_name = sanitize_class_name(class_name)
    if not sanitized_name:
        raise HTTPException(status_code=400, detail="Invalid class name")
    
    # Check if class exists
    class_path = os.path.join("data", "raw", sanitized_name)
    if not os.path.exists(class_path):
        raise HTTPException(status_code=404, detail=f"Class '{sanitized_name}' not found")
    
    # Read file content
    file_content = await file.read()
    
    # Validate image
    is_valid, message = validate_image(file_content)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    # Save file
    filename = file.filename or f"uploaded_{uuid.uuid4().hex[:8]}.jpg"
    file_path = os.path.join(class_path, filename)
    
    # Convert to PIL Image and save
    image = Image.open(io.BytesIO(file_content))
    image.save(file_path)
    
    return {"status": "ok", "filename": filename}

@app.post("/api/capture")
async def capture_image(class_name: str = Form(...), image_base64: str = Form(...)):
    """Save webcam-captured image"""
    sanitized_name = sanitize_class_name(class_name)
    if not sanitized_name:
        raise HTTPException(status_code=400, detail="Invalid class name")
    
    # Check if class exists
    class_path = os.path.join("data", "raw", sanitized_name)
    if not os.path.exists(class_path):
        raise HTTPException(status_code=404, detail=f"Class '{sanitized_name}' not found")
    
    # Decode base64 image
    try:
        header, encoded = image_base64.split(",", 1)
        decoded_image = base64.b64decode(encoded)
    except:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    
    # Validate image
    is_valid, message = validate_image(decoded_image)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    # Save file
    filename = f"webcam_{uuid.uuid4().hex[:8]}.jpg"
    file_path = os.path.join(class_path, filename)
    
    # Convert to PIL Image and save
    image = Image.open(io.BytesIO(decoded_image))
    image.save(file_path)
    
    return {"status": "ok"}

@app.post("/api/train")
async def start_training(request: TrainingRequest):
    """Start training for selected models"""
    task_id = str(uuid.uuid4())
    
    # Validate classes
    raw_data_path = os.path.join("data", "raw")
    if not os.path.exists(raw_data_path):
        raise HTTPException(status_code=400, detail="No classes found")
    
    available_classes = [d for d in os.listdir(raw_data_path) 
                        if os.path.isdir(os.path.join(raw_data_path, d))]
    
    for class_name in request.classes:
        if class_name not in available_classes:
            raise HTTPException(status_code=400, detail=f"Class '{class_name}' not found")
    
    # Validate models
    valid_models = ["logistic", "random_forest", "cnn"]
    for model in request.models:
        if model not in valid_models:
            raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Valid models: {valid_models}")
    
    # Store task
    active_tasks[task_id] = {
        "status": "started",
        "models": request.models,
        "classes": request.classes,
        "progress": 0
    }
    
    # Start training in background
    asyncio.create_task(run_training(task_id, request.models, request.classes, request.hyperparameters or {}))
    
    return {"status": "started", "task_id": task_id}

async def run_training(task_id: str, models: List[str], classes: List[str], hyperparameters: Dict):
    """Run training for specified models"""
    try:
        # Update task status
        active_tasks[task_id]["status"] = "running"
        
        # Validate that we have at least 2 classes for classification
        if len(classes) < 2:
            error_message = f"CRITICAL ERROR: Need at least 2 classes for training, but only {len(classes)} class found. Please create at least one more class. Current classes: {classes}"
            print(error_message)  # Log to console
            raise ValueError(error_message)
        
        # Split data into train/test
        split_data(classes)
        
        # Train models
        trainers = {
            "logistic": LogisticTrainer(),
            "random_forest": RFTrainer(),
            "cnn": CNNTrainer()
        }
        
        results = {}
        for i, model_name in enumerate(models):
            if model_name in trainers:
                try:
                    # Update progress
                    progress = ((i + 1) / len(models)) * 100
                    active_tasks[task_id]["progress"] = progress
                    
                    # Train model
                    trainer = trainers[model_name]
                    model_results = trainer.train(
                        classes=classes,
                        hyperparameters=hyperparameters.get(model_name, {})
                    )
                    results[model_name] = model_results
                    
                    # Update task with intermediate results
                    active_tasks[task_id]["results"] = results
                except Exception as model_error:
                    error_msg = f"Error training {model_name}: {str(model_error)}"
                    print(error_msg)
                    raise Exception(error_msg)
        
        # Update task status
        active_tasks[task_id]["status"] = "completed"
        active_tasks[task_id]["progress"] = 100
        
    except Exception as e:
        active_tasks[task_id]["status"] = "error"
        active_tasks[task_id]["error"] = str(e)
        print(f"Training error: {str(e)}")  # Log the error for debugging
        raise e  # Re-raise the exception for better error propagation

def split_data(classes: List[str], test_size: float = 0.2):
    """Split raw data into train/test sets"""
    try:
        # Validate input
        if not classes:
            raise ValueError("No classes provided for data splitting")
            
        if len(classes) < 2:
            raise ValueError(f"Need at least 2 classes for classification, but only {len(classes)} class found. Please create at least one more class.")
        
        processed_path = os.path.join("data", "processed")
        train_path = os.path.join(processed_path, "train")
        test_path = os.path.join(processed_path, "test")
        
        # Clear existing processed data
        if os.path.exists(train_path):
            shutil.rmtree(train_path)
        if os.path.exists(test_path):
            shutil.rmtree(test_path)
        
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        
        # Create class directories
        for class_name in classes:
            os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_path, class_name), exist_ok=True)
        
        # Split data for each class
        import random
        random.seed(42)
        
        total_images = 0
        for class_name in classes:
            class_raw_path = os.path.join("data", "raw", class_name)
            if not os.path.exists(class_raw_path):
                continue
                
            # Get all images
            images = [f for f in os.listdir(class_raw_path) 
                     if os.path.isfile(os.path.join(class_raw_path, f)) and 
                     f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(images) == 0:
                print(f"Warning: No images found in class '{class_name}'")
                continue
                
            total_images += len(images)
            
            # Shuffle and split
            random.shuffle(images)
            split_idx = int(len(images) * (1 - test_size))
            # Ensure at least one image in each set if possible
            if len(images) > 1 and split_idx == len(images):
                split_idx = len(images) - 1
            elif split_idx == 0 and len(images) > 1:
                split_idx = 1
                
            train_images = images[:split_idx] if split_idx > 0 else images[:1]
            test_images = images[split_idx:] if split_idx < len(images) else images[1:]
            
            # Copy to respective directories
            for img in train_images:
                src = os.path.join(class_raw_path, img)
                dst = os.path.join(train_path, class_name, img)
                shutil.copy2(src, dst)
                
            for img in test_images:
                src = os.path.join(class_raw_path, img)
                dst = os.path.join(test_path, class_name, img)
                shutil.copy2(src, dst)
        
        if total_images == 0:
            raise ValueError("No images found in any of the selected classes")
            
        print(f"Data split completed. Total images processed: {total_images}")
        
    except Exception as e:
        print(f"Error in split_data: {str(e)}")
        raise e

@app.get("/api/train/status/{task_id}")
async def get_training_status(task_id: str):
    """Get training status"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    return task

@app.websocket("/ws/train/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for realtime training updates"""
    if task_id not in active_tasks:
        await websocket.close(code=4004)
        return
    
    await websocket.accept()
    
    try:
        # Send updates while training is running
        last_progress = -1
        while True:
            if task_id not in active_tasks:
                break
                
            task = active_tasks[task_id]
            progress = task.get("progress", 0)
            
            # Only send updates when progress changes
            if progress != last_progress:
                await websocket.send_json({
                    "status": task["status"],
                    "progress": progress
                })
                last_progress = progress
            
            # Check if training is completed
            if task["status"] in ["completed", "error"]:
                await websocket.send_json({
                    "status": task["status"],
                    "progress": 100 if task["status"] == "completed" else -1,
                    "results": task.get("results", {}),
                    "error": task.get("error", None)
                })
                break
            
            # Wait before sending next update
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.get("/api/evaluate/{model_name}")
async def evaluate_model(model_name: str):
    """Return evaluation metrics for a model"""
    # Implementation would load the trained model and evaluate it
    # For now, returning placeholder data
    return {
        "accuracy": 0.95,
        "confusion_matrix": [[45, 3], [2, 48]],
        "precision": [0.96, 0.94],
        "recall": [0.94, 0.96],
        "f1": [0.95, 0.95]
    }

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict on a single image"""
    # Read file content
    file_content = await file.read()
    
    # Validate image
    is_valid, message = validate_image(file_content)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    # Make predictions
    predictor = Predictor()
    results = predictor.predict(file_content)
    
    return results

@app.post("/api/predict/webcam")
async def predict_webcam(image_base64: str = Form(...)):
    """Predict from base64 frame"""
    # Decode base64 image
    try:
        header, encoded = image_base64.split(",", 1)
        decoded_image = base64.b64decode(encoded)
    except:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    
    # Validate image
    is_valid, message = validate_image(decoded_image)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    # Make predictions
    predictor = Predictor()
    results = predictor.predict(decoded_image)
    
    return results

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)