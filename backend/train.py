#!/usr/bin/env python3
"""
Deepfake Detection Model Training Script
Uses Xception architecture with transfer learning
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
MODEL_PATH = Path(__file__).parent / 'model' / 'deepfake_detector.h5'

def create_model():
    """
    Create Xception-based deepfake detection model
    Uses transfer learning with frozen base layers
    """
    # Load pre-trained Xception (without top layers)
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer: sigmoid for binary classification
    output = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def create_sample_dataset():
    """
    Create a small synthetic dataset for demonstration
    In production, replace with actual deepfake dataset from Kaggle
    """
    dataset_path = Path(__file__).parent / 'dataset'
    real_path = dataset_path / 'real'
    fake_path = dataset_path / 'fake'
    
    # Create directories
    real_path.mkdir(parents=True, exist_ok=True)
    fake_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Dataset directory created at: {dataset_path}")
    print("⚠️  NOTE: For production, download real deepfake dataset from Kaggle")
    print("   Suggested: 'Deepfake Detection Challenge' or 'Real vs Fake Faces'")
    
    # Create dummy images if none exist
    if len(list(real_path.glob('*.jpg'))) == 0:
        print("\n🎨 Creating sample images for demonstration...")
        for i in range(50):
            # Create random images as placeholders
            img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            tf.keras.preprocessing.image.save_img(
                real_path / f'real_{i}.jpg', img
            )
            
        for i in range(50):
            img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            tf.keras.preprocessing.image.save_img(
                fake_path / f'fake_{i}.jpg', img
            )
    
    return dataset_path

def prepare_data_generators(dataset_path):
    """
    Create data generators with augmentation
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def plot_training_history(history, save_path):
    """
    Plot training and validation metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n📊 Training plots saved to: {save_path}")

def evaluate_model(model, val_generator):
    """
    Evaluate model and generate confusion matrix
    """
    print("\n🔍 Evaluating model...")
    
    # Get predictions
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=1)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = val_generator.classes
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = Path(__file__).parent / 'model' / 'confusion_matrix.png'
    plt.savefig(cm_path)
    print(f"📊 Confusion matrix saved to: {cm_path}")

def train():
    """
    Main training function
    """
    print("🚀 Starting Deepfake Detection Model Training")
    print("=" * 50)
    
    # Create model directory
    model_dir = Path(__file__).parent / 'model'
    model_dir.mkdir(exist_ok=True)
    
    # Prepare dataset
    print("\n📁 Preparing dataset...")
    dataset_path = create_sample_dataset()
    
    # Create data generators
    print("\n🔄 Creating data generators...")
    train_gen, val_gen = prepare_data_generators(dataset_path)
    
    print(f"\n✅ Found {train_gen.samples} training images")
    print(f"✅ Found {val_gen.samples} validation images")
    print(f"Classes: {train_gen.class_indices}")
    
    # Create model
    print("\n🏗️  Building Xception model...")
    model = create_model()
    print(model.summary())
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n🎯 Training for {EPOCHS} epochs...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_path = model_dir / 'training_history.png'
    plot_training_history(history, plot_path)
    
    # Evaluate model
    evaluate_model(model, val_gen)
    
    print("\n✅ Training completed successfully!")
    print(f"📦 Model saved at: {MODEL_PATH}")
    print("\n💡 Next steps:")
    print("   1. Start the backend server")
    print("   2. Use the web interface to test predictions")
    print("   3. For production: Replace sample data with real deepfake dataset")

if __name__ == '__main__':
    train()
