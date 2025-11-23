import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from PIL import Image
import json
from typing import List, Dict, Optional, Any
import cv2

class RFTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.image_size = (224, 224)
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for random forest"""
        # Load image
        image = Image.open(image_path)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Resize
        image = image.resize(self.image_size)
        # Convert to array and flatten
        image_array = np.array(image)
        flattened = image_array.flatten()
        # Normalize
        normalized = flattened / 255.0
        return normalized
    
    def load_dataset(self, classes: List[str]) -> tuple:
        """Load and preprocess dataset"""
        # Validate input
        if not classes:
            raise ValueError("No classes provided for training")
            
        if len(classes) < 2:
            raise ValueError(f"Need at least 2 classes for classification, but only {len(classes)} class found. Please create at least one more class.")
        
        X = []
        y = []
        
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join("data", "processed", "train", class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Class directory '{class_path}' does not exist")
                continue
                
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                print(f"Warning: No images found in class '{class_name}'")
                continue
                
            for filename in image_files:
                image_path = os.path.join(class_path, filename)
                try:
                    # Preprocess image
                    processed_image = self.preprocess_image(image_path)
                    X.append(processed_image)
                    y.append(class_idx)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
        
        if len(X) == 0:
            raise ValueError("No valid training data found. Please ensure you have at least 10 images per class.")
            
        return np.array(X), np.array(y)
    
    def train(self, classes: List[str], hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Train random forest model"""
        if hyperparameters is None:
            hyperparameters = {}
            
        try:
            # Validate input
            if not classes:
                raise ValueError("No classes provided for training")
                
            if len(classes) < 2:
                raise ValueError(f"Need at least 2 classes for classification, but only {len(classes)} class found. Please create at least one more class.")
            
            # Load dataset
            X, y = self.load_dataset(classes)
            
            if len(X) == 0:
                raise ValueError("No training data found")
            
            # Check if we have enough data for train/test split
            if len(X) < 2:
                raise ValueError("Need at least 2 images to perform train/test split")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create model
            n_estimators = hyperparameters.get('n_estimators', 200)
            
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                n_jobs=-1,
                random_state=42
            )
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_accuracy = self.model.score(X_train_scaled, y_train)
            test_accuracy = self.model.score(X_test_scaled, y_test)
            
            # Save model
            model_path = os.path.join("models", "random_forest.pkl")
            joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = os.path.join("models", "rf_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            
            # Save training log
            log_data = {
                "model": "random_forest",
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "n_estimators": n_estimators,
                "num_samples": len(X),
                "num_features": X.shape[1] if len(X) > 0 else 0
            }
            
            log_path = os.path.join("logs", "rf_training_log.json")
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            return {
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy)
            }
        except Exception as e:
            print(f"Error in RandomForest training: {str(e)}")
            raise e