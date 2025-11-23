import os
import sys
import json
from trainers.logistic_trainer import LogisticTrainer
from trainers.rf_trainer import RFTrainer
from trainers.cnn_trainer import CNNTrainer

def train_all_models():
    """Train all models with proper error handling"""
    print("Starting model training...")
    
    # Get classes
    processed_path = os.path.join("data", "processed", "train")
    if not os.path.exists(processed_path):
        print("Error: Processed data not found. Please create classes and add images first.")
        return False
        
    classes = sorted([d for d in os.listdir(processed_path) 
                     if os.path.isdir(os.path.join(processed_path, d))])
    
    if len(classes) < 2:
        print(f"Error: Need at least 2 classes for training, but only {len(classes)} found: {classes}")
        return False
        
    print(f"Found classes: {classes}")
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    try:
        logistic_trainer = LogisticTrainer()
        logistic_results = logistic_trainer.train(classes=classes)
        print(f"Logistic Regression training completed. Train accuracy: {logistic_results['train_accuracy']:.4f}, Test accuracy: {logistic_results['test_accuracy']:.4f}")
    except Exception as e:
        print(f"Error training Logistic Regression: {e}")
        
    # Train Random Forest
    print("\nTraining Random Forest...")
    try:
        rf_trainer = RFTrainer()
        rf_results = rf_trainer.train(classes=classes)
        print(f"Random Forest training completed. Train accuracy: {rf_results['train_accuracy']:.4f}, Test accuracy: {rf_results['test_accuracy']:.4f}")
    except Exception as e:
        print(f"Error training Random Forest: {e}")
        
    # Train CNN
    print("\nTraining CNN...")
    try:
        cnn_trainer = CNNTrainer()
        cnn_results = cnn_trainer.train(classes=classes)
        print(f"CNN training completed. Train accuracy: {cnn_results['train_accuracy']:.4f}, Validation accuracy: {cnn_results['val_accuracy']:.4f}")
    except Exception as e:
        print(f"Error training CNN: {e}")
        
    print("\nModel training completed!")
    return True

if __name__ == "__main__":
    success = train_all_models()
    sys.exit(0 if success else 1)