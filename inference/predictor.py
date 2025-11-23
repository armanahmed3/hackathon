import os
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf
import io
from typing import Dict, List
import json

class Predictor:
    # Class-level cache for models to avoid reloading
    _models_loaded = False
    _logistic_model = None
    _rf_model = None
    _cnn_model = None
    _logistic_scaler = None
    _rf_scaler = None
    _classes = []
    
    def __init__(self):
        self.logistic_model = None
        self.rf_model = None
        self.cnn_model = None
        self.logistic_scaler = None
        self.rf_scaler = None
        self.classes = []
        self.load_models()
        self.load_classes()
        
    def load_models(self):
        """Load trained models"""
        # Use cached models if already loaded
        if Predictor._models_loaded:
            self.logistic_model = Predictor._logistic_model
            self.rf_model = Predictor._rf_model
            self.cnn_model = Predictor._cnn_model
            self.logistic_scaler = Predictor._logistic_scaler
            self.rf_scaler = Predictor._rf_scaler
            return
            
        try:
            # Load logistic regression model
            if os.path.exists("models/logistic.pkl"):
                self.logistic_model = joblib.load("models/logistic.pkl")
                Predictor._logistic_model = self.logistic_model
            
            # Load random forest model
            if os.path.exists("models/random_forest.pkl"):
                self.rf_model = joblib.load("models/random_forest.pkl")
                Predictor._rf_model = self.rf_model
            
            # Load CNN model
            if os.path.exists("models/cnn_model.h5"):
                self.cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
                Predictor._cnn_model = self.cnn_model
                
            # Load scalers
            if os.path.exists("models/logistic_scaler.pkl"):
                self.logistic_scaler = joblib.load("models/logistic_scaler.pkl")
                Predictor._logistic_scaler = self.logistic_scaler
                
            if os.path.exists("models/rf_scaler.pkl"):
                self.rf_scaler = joblib.load("models/rf_scaler.pkl")
                Predictor._rf_scaler = self.rf_scaler
                
            # Mark models as loaded
            Predictor._models_loaded = True
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def load_classes(self):
        """Load class names from processed data"""
        # Use cached classes if already loaded
        if Predictor._classes:
            self.classes = Predictor._classes
            return
            
        processed_path = os.path.join("data", "processed", "train")
        if os.path.exists(processed_path):
            self.classes = sorted([d for d in os.listdir(processed_path) 
                                 if os.path.isdir(os.path.join(processed_path, d))])
            Predictor._classes = self.classes
    
    def preprocess_image_for_classical(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for classical ML models (logistic, random forest)"""
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Resize
        image = image.resize((224, 224))
        # Convert to array and flatten
        image_array = np.array(image)
        flattened = image_array.flatten()
        # Normalize
        normalized = flattened / 255.0
        return normalized.reshape(1, -1)  # Reshape for single sample
    
    def preprocess_image_for_cnn(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for CNN model"""
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Resize
        image = image.resize((224, 224))
        # Convert to array and normalize
        image_array = np.array(image) / 255.0
        # Add batch dimension
        return np.expand_dims(image_array, axis=0)
    
    def predict(self, image_bytes: bytes) -> Dict:
        """Make predictions using all available models"""
        results = {}
        probabilities = {}
        
        # Predict with logistic regression
        if self.logistic_model is not None and self.logistic_scaler is not None:
            try:
                processed_image = self.preprocess_image_for_classical(image_bytes)
                scaled_image = self.logistic_scaler.transform(processed_image)
                logistic_pred = self.logistic_model.predict(scaled_image)[0]
                logistic_proba = self.logistic_model.predict_proba(scaled_image)[0]
                
                results["logistic_regression"] = self.classes[logistic_pred]
                probabilities["logistic"] = {
                    self.classes[i]: float(prob) for i, prob in enumerate(logistic_proba)
                }
            except Exception as e:
                print(f"Error in logistic regression prediction: {e}")
                results["logistic_regression"] = "Error"
                probabilities["logistic"] = {}
        
        # Predict with random forest
        if self.rf_model is not None and self.rf_scaler is not None:
            try:
                processed_image = self.preprocess_image_for_classical(image_bytes)
                scaled_image = self.rf_scaler.transform(processed_image)
                rf_pred = self.rf_model.predict(scaled_image)[0]
                rf_proba = self.rf_model.predict_proba(scaled_image)[0]
                
                results["random_forest"] = self.classes[rf_pred]
                probabilities["random_forest"] = {
                    self.classes[i]: float(prob) for i, prob in enumerate(rf_proba)
                }
            except Exception as e:
                print(f"Error in random forest prediction: {e}")
                results["random_forest"] = "Error"
                probabilities["random_forest"] = {}
        
        # Predict with CNN
        if self.cnn_model is not None:
            try:
                processed_image = self.preprocess_image_for_cnn(image_bytes)
                cnn_pred = self.cnn_model.predict(processed_image, verbose=0)
                predicted_class_idx = np.argmax(cnn_pred)
                predicted_class = self.classes[predicted_class_idx]
                
                results["cnn"] = predicted_class
                probabilities["cnn"] = {
                    self.classes[i]: float(prob) for i, prob in enumerate(cnn_pred[0])
                }
            except Exception as e:
                print(f"Error in CNN prediction: {e}")
                results["cnn"] = "Error"
                probabilities["cnn"] = {}
        
        # Add probabilities to results
        results["probabilities"] = probabilities
        
        return results