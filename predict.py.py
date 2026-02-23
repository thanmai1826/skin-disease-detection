import numpy as np
from tensorflow.keras.models import load_model
import sys
sys.path.append('..')
from src.preprocess import preprocess_image, CLASSES

# Load the trained model globally
MODEL_PATH = 'models/skin_model.h5'
model = None

def load_trained_model():
    global model
    if model is None:
        print("Loading model...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    return model

def predict_disease(image_path):
    """
    Accepts an image path, preprocesses it, and returns prediction.
    """
    # Load model
    loaded_model = load_trained_model()
    
    # Preprocess image
    processed_image = preprocess_image(image_path)
    
    # Predict
    predictions = loaded_model.predict(processed_image)
    
    # Get the index of the highest probability
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    predicted_class = CLASSES[predicted_class_index]
    
    return predicted_class, float(confidence)

if __name__ == "__main__":
    # Example usage:
    # Replace 'path_to_image.jpg' with a real path
    # class_name, conf = predict_disease('dataset/acne/image.jpg')
    # print(f"Prediction: {class_name}, Confidence: {conf:.2f}")
    pass