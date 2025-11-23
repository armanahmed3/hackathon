import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
from typing import List, Dict, Optional, Any
import cv2
from PIL import Image

class CNNTrainer:
    def __init__(self):
        self.model = None
        self.image_size = (224, 224)
        self.channels = 3
        
    def build_model(self, num_classes: int) -> tf.keras.Model:
        """Build CNN model architecture"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, self.channels)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def load_dataset(self, classes: List[str]) -> tuple:
        """Load and preprocess dataset"""
        try:
            # Check if data directory exists
            train_dir = 'data/processed/train'
            if not os.path.exists(train_dir):
                raise ValueError(f"Training directory '{train_dir}' does not exist")
            
            # Check if classes exist
            for class_name in classes:
                class_path = os.path.join(train_dir, class_name)
                if not os.path.exists(class_path):
                    print(f"Warning: Class directory '{class_path}' does not exist")
            
            # Use ImageDataGenerator for automatic preprocessing and augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                validation_split=0.2
            )
            
            train_generator = train_datagen.flow_from_directory(
                'data/processed/train',
                target_size=self.image_size,
                batch_size=32,
                class_mode='categorical',
                classes=classes,
                subset='training'
            )
            
            validation_generator = train_datagen.flow_from_directory(
                'data/processed/train',
                target_size=self.image_size,
                batch_size=32,
                class_mode='categorical',
                classes=classes,
                subset='validation'
            )
            
            return train_generator, validation_generator
        except Exception as e:
            print(f"Error in load_dataset: {str(e)}")
            raise e
    
    def train(self, classes: List[str], hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Train CNN model"""
        if hyperparameters is None:
            hyperparameters = {}
            
        try:
            # Validate input
            if not classes:
                raise ValueError("No classes provided for training")
                
            if len(classes) < 2:
                raise ValueError(f"Need at least 2 classes for classification, but only {len(classes)} class found. Please create at least one more class.")
            
            num_classes = len(classes)
            if num_classes == 0:
                raise ValueError("No classes provided for training")
            
            # Build model
            self.model = self.build_model(num_classes)
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Load dataset
            train_gen, val_gen = self.load_dataset(classes)
            
            # Check if we have data
            if train_gen.samples == 0:
                raise ValueError("No training data found. Please ensure you have at least 10 images per class.")
            
            # Set hyperparameters
            epochs = hyperparameters.get('epochs', 20)
            batch_size = hyperparameters.get('batch_size', 32)
            
            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            model_checkpoint = ModelCheckpoint(
                'models/cnn_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
            
            # Train model
            history = self.model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=[early_stopping, model_checkpoint],
                verbose='auto'
            )
            
            # Save final model
            model_path = os.path.join("models", "cnn_model.h5")
            self.model.save(model_path)
            
            # Save training log
            # Extract history data safely
            if history is not None and hasattr(history, 'history'):
                hist = history.history
                train_acc = float(hist['accuracy'][-1]) if 'accuracy' in hist and hist['accuracy'] else 0.0
                val_acc = float(hist['val_accuracy'][-1]) if 'val_accuracy' in hist and hist['val_accuracy'] else 0.0
                train_loss = float(hist['loss'][-1]) if 'loss' in hist and hist['loss'] else 0.0
                val_loss = float(hist['val_loss'][-1]) if 'val_loss' in hist and hist['val_loss'] else 0.0
            else:
                train_acc = val_acc = train_loss = val_loss = 0.0
            
            log_data = {
                "model": "cnn",
                "epochs": epochs,
                "batch_size": batch_size,
                "num_classes": num_classes,
                "final_train_accuracy": train_acc,
                "final_val_accuracy": val_acc,
                "final_train_loss": train_loss,
                "final_val_loss": val_loss
            }
            
            log_path = os.path.join("logs", "cnn_training_log.json")
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            return {
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
        except Exception as e:
            print(f"Error in CNN training: {str(e)}")
            raise e