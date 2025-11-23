import gradio as gr
import os
import numpy as np
from PIL import Image
import io
import json
from inference.predictor import Predictor

# Initialize predictor
predictor = Predictor()

def predict_image(image):
    """Make prediction on uploaded image"""
    try:
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Make prediction
        results = predictor.predict(img_byte_arr)
        
        # Format results for display
        formatted_results = []
        
        if results.get("logistic_regression") and results["logistic_regression"] != "Error":
            formatted_results.append(("Logistic Regression", results["logistic_regression"]))
            
        if results.get("random_forest") and results["random_forest"] != "Error":
            formatted_results.append(("Random Forest", results["random_forest"]))
            
        if results.get("cnn") and results["cnn"] != "Error":
            formatted_results.append(("CNN", results["cnn"]))
        
        # Create a formatted string for display
        result_text = ""
        for model, prediction in formatted_results:
            result_text += f"{model}: {prediction}\n"
            
        if not result_text:
            result_text = "No predictions available. Please train models first."
            
        return result_text
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Teachable Machine Classifier") as demo:
    gr.Markdown("# Teachable Machine Image Classifier")
    gr.Markdown("Upload an image to classify it using trained models")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Classify Image")
        with gr.Column():
            text_output = gr.Textbox(label="Prediction Results", lines=10)
    
    submit_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=text_output
    )
    
    gr.Markdown("## How to Use")
    gr.Markdown("""
    1. Upload an image using the interface above
    2. Click 'Classify Image' button
    3. View predictions from available models
    
    Note: Models must be trained before predictions can be made.
    """)
    
    gr.Markdown("## Model Information")
    gr.Markdown("""
    - **Logistic Regression**: Traditional ML approach
    - **Random Forest**: Ensemble learning method
    - **CNN**: Deep learning convolutional neural network
    """)

if __name__ == "__main__":
    demo.launch()