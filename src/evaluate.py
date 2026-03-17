'''"""
Model Evaluation Script for Driver Drowsiness Detection
Generates classification report, confusion matrix, and per-class metrics.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration
MODEL_PATH = './models/best_model.keras'
TEST_DIR = './data/test'
RESULTS_DIR = './results'

IMG_SIZE = 224
BATCH_SIZE = 32
CLASS_NAMES = ['Closed_Eyes', 'No_Yawn', 'Open_Eyes', 'Yawn']

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model():
    """Load trained model."""
    print("Loading model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def create_test_generator():
    """Create test data generator."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=CLASS_NAMES
    )
    
    return test_gen


def evaluate_model(model, test_gen):
    """Evaluate model and return metrics."""
    print(f"\\nEvaluating on {test_gen.samples} test images...")
    
    # Overall metrics
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    
    # Predictions
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    return loss, accuracy, y_true, y_pred, predictions


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\\nConfusion matrix saved: {save_path}")
    plt.close()
    
    return cm


def print_results(accuracy, y_true, y_pred, class_names):
    """Print detailed evaluation results."""
    print("\\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Correct Predictions: {int(accuracy * len(y_true))} / {len(y_true)}")
    
    print("\\nClassification Report:")
    print("-"*60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Per-class accuracy
    print("Per-Class Accuracy:")
    print("-"*60)
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
            correct = np.sum((y_true == i) & (y_pred == i))
            total = np.sum(y_true == i)
            print(f"{class_name:15}: {class_acc*100:6.2f}% ({correct}/{total})")


def save_report(accuracy, report_text):
    """Save evaluation report to file."""
    report_path = os.path.join(RESULTS_DIR, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Driver Drowsiness Detection - Model Evaluation\\n")
        f.write("="*60 + "\\n\\n")
        f.write(f"Model: {MODEL_PATH}\\n")
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\\n\\n")
        f.write(report_text)
    
    print(f"\\nReport saved: {report_path}")


def main():
    """Main evaluation function."""
    print("="*60)
    print("DROWSINESS DETECTION - MODEL EVALUATION")
    print("="*60)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Create generator
    test_gen = create_test_generator()
    print(f"Found {test_gen.samples} test images")
    print(f"Classes: {CLASS_NAMES}")
    
    # Evaluate
    loss, accuracy, y_true, y_pred, predictions = evaluate_model(model, test_gen)
    
    # Print results
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print_results(accuracy, y_true, y_pred, CLASS_NAMES)
    
    # Generate confusion matrix
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)
    
    # Save report
    save_report(accuracy, report)
    
    print("\\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
'''
