'''"""
CNN Training Script for Driver Drowsiness Detection
Trains a 4-class classifier (Closed_Eyes, Open_Eyes, No_Yawn, Yawn)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

DATA_DIR = "./data/train"  # Change to your dataset path
VAL_DIR = "./data/val"     # Set to None if using validation_split
MODEL_DIR = "./models"

CLASS_NAMES = ['Closed_Eyes', 'No_Yawn', 'Open_Eyes', 'Yawn']
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

# Set seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"Classes: {CLASS_NAMES}\\n")


def create_data_generators():
    """Create training and validation data generators."""
    
    # Training augmentation - mild to preserve features
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        zoom_range=0.10,
        horizontal_flip=True,
        brightness_range=[0.90, 1.10],
        fill_mode='nearest'
    )
    
    # Validation - only rescaling
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True,
        seed=SEED
    )
    
    # Validation generator
    if VAL_DIR and os.path.exists(VAL_DIR):
        val_gen = val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=CLASS_NAMES,
            shuffle=False,
            seed=SEED
        )
    else:
        # Use 20% of training data for validation
        print("No separate validation folder found. Using 20% split from training data.")
        val_gen = train_datagen.flow_from_directory(
            DATA_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=CLASS_NAMES,
            subset='validation',
            shuffle=False,
            seed=SEED
        )
    
    return train_gen, val_gen


def build_model():
    """Build CNN architecture."""
    
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Block 2
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Block 3
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks():
    """Define training callbacks."""
    
    return [
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]


def train_model():
    """Main training function."""
    
    print("Loading data...")
    train_gen, val_gen = create_data_generators()
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Class mapping: {train_gen.class_indices}\\n")
    
    print("Building model...")
    model = build_model()
    model.summary()
    
    print("\\nStarting training...\\n")
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=1
    )
    
    # Save final model
    model.save(FINAL_MODEL_PATH)
    print(f"\\nFinal model saved: {FINAL_MODEL_PATH}")
    print(f"Best model saved: {BEST_MODEL_PATH}")
    
    # Final evaluation
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\\nFinal validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Final validation loss: {val_loss:.4f}")
    
    return history


if __name__ == "__main__":
    train_model()
'''
