import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define dataset path
DATASET_PATH = 'dataset'
# Image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
# Classes
CLASSES = ['acne', 'eczema', 'psoriasis', 'melanoma', 'normal']

def load_data():
    """
    Loads images from the dataset folder, resizes them, 
    and creates label arrays.
    """
    data = []
    labels = []

    print("Loading data...")

    for class_name in CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} not found. Skipping.")
            continue

        # Iterate through images in the folder
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            # Read image using OpenCV (BGR format)
            image = cv2.imread(img_path)
            
            # Check if image is loaded correctly
            if image is not None:
                # Resize image
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                # Convert BGR to RGB (OpenCV loads as BGR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Normalize pixel values (0-255 -> 0-1)
                image = image / 255.0
                
                data.append(image)
                labels.append(CLASSES.index(class_name))
            else:
                print(f"Warning: Could not load image {img_path}")

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # One-hot encode labels (e.g., 1 -> [0, 1, 0, 0, 0])
    labels = to_categorical(labels, num_classes=len(CLASSES))

    print(f"Total images loaded: {len(data)}")
    
    return data, labels

def preprocess_image(image_path):
    """
    Preprocesses a single image for prediction.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    # Add batch dimension -> (1, 128, 128, 3)
    image = np.expand_dims(image, axis=0)
    return image

if __name__ == "__main__":
    # Quick test to see data shape
    X, y = load_data()
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")