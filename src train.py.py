import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Import from sibling module
import sys
sys.path.append('..')
from src.preprocess import load_data, CLASSES

# Create models directory if not exists
if not os.path.exists('models'):
    os.makedirs('models')

def build_model():
    """
    Builds the CNN architecture.
    """
    model = Sequential([
        # Conv Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        
        # Conv Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Conv Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Fully Connected Layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5), # Prevents overfitting
        Dense(len(CLASSES), activation='softmax') # Output layer
    ])

    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def plot_training_history(history):
    """
    Plots accuracy and loss graphs.
    """
    # Plot Accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_plot.png')
    print("Training plot saved as training_plot.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """
    Plots Confusion Matrix.
    """
    # Convert one-hot encoded labels back to integers
    y_true_int = np.argmax(y_true, axis=1)
    y_pred_int = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true_int, y_pred_int)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved as confusion_matrix.png")
    plt.close()

def main():
    # 1. Load Data
    X, y = load_data()
    
    # 2. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    # 3. Build Model
    model = build_model()

    # 4. Train Model
    # Note: Adjust epochs (e.g., 15) based on your dataset size
    history = model.fit(
        X_train, y_train,
        epochs=15,
        validation_split=0.1, # Use 10% of training data for validation
        batch_size=32
    )

    # 5. Evaluate Model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # 6. Visualizations
    plot_training_history(history)
    
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=CLASSES))

    # 7. Save Model
    model.save('models/skin_model.h5')
    print("Model saved to models/skin_model.h5")

if __name__ == "__main__":
    main()