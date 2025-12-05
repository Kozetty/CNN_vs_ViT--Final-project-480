"""
Brain Tumor Detection: CNN vs Vision Transformer (ViT) Performance Comparison
Python script for VS Code or any Python IDE
"""

import os
import warnings
import itertools
import cv2
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

# General parameters
EPOCHS_CNN = 15  # You can change this to 100 for full training
EPOCHS_VIT = 15  # You can change this to 100 for full training
PIC_SIZE = 240

# Dataset path - UPDATE THIS TO YOUR DATASET LOCATION
FOLDER_PATH = "."
# Example: FOLDER_PATH = "C:/Users/YourName/Desktop/brain_tumor_dataset"

# ViT hyperparameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 256
IMAGE_SIZE = 240
PATCH_SIZE = 20
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
TRANSFORMER_LAYERS = 8
MLP_HEAD_UNITS = [2048, 1024]

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_dataset(folder_path, pic_size):
    """Load and preprocess the brain tumor dataset"""
    print("Loading dataset...")
    no_images = os.listdir(folder_path + '/no/')
    yes_images = os.listdir(folder_path + '/yes/')
    dataset = []
    lab = []

    # Load images without tumor
    for image_name in no_images:
        image = cv2.imread(folder_path + '/no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((pic_size, pic_size))
        dataset.append(np.array(image))
        lab.append(0)

    # Load images with tumor
    for image_name in yes_images:
        image = cv2.imread(folder_path + '/yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((pic_size, pic_size))
        dataset.append(np.array(image))
        lab.append(1)

    # Convert to numpy arrays
    dataset = np.array(dataset)
    lab = np.array(lab)
    print(f"Dataset shape: {dataset.shape}, Labels shape: {lab.shape}")
    return dataset, lab

def plot_sample_images(folder_path, state, pic_size, save_path=None):
    """Plot sample images from the dataset"""
    plt.figure(figsize=(12, 12))
    for i in range(1, 10, 1):
        plt.subplot(3, 3, i)
        img = load_img(folder_path + '/' + state + '/' + os.listdir(folder_path + '/' + state)[i], 
                      target_size=(pic_size, pic_size))
        plt.imshow(img)
    plt.suptitle(f'Sample Images - {state}')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

# ============================================================================
# CNN MODEL
# ============================================================================

def create_cnn_model(pic_size):
    """Create CNN model architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), 
                               activation="relu", padding="valid", 
                               input_shape=(pic_size, pic_size, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), 
                               activation="relu", padding="valid"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation='relu',
                             kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3),
                             bias_regularizer=regularizers.L2(1e-2),
                             activity_regularizer=regularizers.L2(1e-3)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),
    ])
    return model

def plot_training_history(history, title, save_path=None):
    """Plot training and validation loss and accuracy"""
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    plt.suptitle(title, fontsize=14)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_confusion_matrix(cm, classes, title='Confusion matrix', 
                         cmap=plt.cm.Blues, save_path=None):
    """Plot normalized confusion matrix"""
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

# ============================================================================
# VISION TRANSFORMER (ViT) MODEL
# ============================================================================

def mlp(x, hidden_units, dropout_rate):
    """Multi-layer perceptron"""
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    """Layer to extract patches from images"""
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    """Encodes patches with linear projection and position embeddings"""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection.units,
        })
        return config

def create_data_augmentation(image_size, x_train):
    """Create data augmentation pipeline for ViT"""
    data_augmentation = tf.keras.Sequential([
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ], name="data_augmentation")
    
    # Compute mean and variance for normalization
    data_augmentation.layers[0].adapt(x_train)
    return data_augmentation

def create_vit_classifier(image_size, patch_size, num_patches, projection_dim,
                         transformer_layers, num_heads, transformer_units,
                         mlp_head_units, data_augmentation):
    """Create Vision Transformer classifier"""
    inputs = layers.Input(shape=(240, 240, 3))
    # Augment data
    augmented = data_augmentation(inputs)
    # Create patches
    patches = Patches(patch_size)(augmented)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # Create multiple layers of the Transformer block
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])
    
    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs
    logits = layers.Dense(2)(features)
    # Create the Keras model
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("Brain Tumor Detection: CNN vs Vision Transformer Comparison")
    print("=" * 80)
    
    # Load dataset
    dataset, lab = load_dataset(FOLDER_PATH, PIC_SIZE)
    
    # Train-test split
    print("\nSplitting dataset into train and test sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        dataset, lab, test_size=0.2, shuffle=True, random_state=42
    )
    print(f"Training set: {x_train.shape}, Test set: {x_test.shape}")
    
    # Visualize sample images
    print("\nVisualizing sample images...")
    plot_sample_images(FOLDER_PATH, 'yes', PIC_SIZE, 'sample_tumor_yes.png')
    plot_sample_images(FOLDER_PATH, 'no', PIC_SIZE, 'sample_tumor_no.png')
    
    # ========================================================================
    # CNN MODEL TRAINING
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING CNN MODEL")
    print("=" * 80)
    
    # Create and compile CNN model
    print("\nCreating CNN model...")
    cnn_model = create_cnn_model(PIC_SIZE)
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', 
                     metrics=['accuracy'])
    cnn_model.summary()
    
    # Calculate class weights
    print("\nCalculating class weights...")
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), class_weights))
    print(f"Class weights: {class_weights}")
    
    # Train CNN model
    print(f"\nTraining CNN model for {EPOCHS_CNN} epochs...")
    cnn_history = cnn_model.fit(
        x_train, y_train, 
        epochs=EPOCHS_CNN, 
        class_weight=class_weights, 
        validation_data=(x_test, y_test), 
        verbose=1
    )
    
    # Evaluate CNN model
    print("\nEvaluating CNN model...")
    cnn_results = cnn_model.evaluate(x_test, y_test)
    print(f'CNN Model Accuracy: {round(cnn_results[1]*100, 2)}%')
    
    # Plot CNN training history
    plot_training_history(cnn_history, 'CNN Training History - Optimizer: Adam',
                         'cnn_training_history.png')
    
    # CNN predictions and confusion matrix
    print("\nGenerating CNN predictions...")
    cnn_predictions = cnn_model.predict(x_test)
    cnn_y_pred = [1 if i >= 0.5 else 0 for i in cnn_predictions]
    
    cnn_cnf_matrix = confusion_matrix(y_test, cnn_y_pred)
    plot_confusion_matrix(cnn_cnf_matrix, classes=["Yes", "No"], 
                         title='CNN Normalized Confusion Matrix',
                         save_path='cnn_confusion_matrix.png')
    
    print("\nCNN Classification Report:")
    print(classification_report(y_test, cnn_y_pred, target_names=["No Tumor", "Tumor"]))
    
    # ========================================================================
    # VISION TRANSFORMER MODEL TRAINING
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING VISION TRANSFORMER (ViT) MODEL")
    print("=" * 80)
    
    # Create data augmentation
    print("\nCreating data augmentation pipeline...")
    data_augmentation = create_data_augmentation(IMAGE_SIZE, x_train)
    
    # Create ViT model
    print("\nCreating Vision Transformer model...")
    vit_classifier = create_vit_classifier(
        IMAGE_SIZE, PATCH_SIZE, NUM_PATCHES, PROJECTION_DIM,
        TRANSFORMER_LAYERS, NUM_HEADS, TRANSFORMER_UNITS,
        MLP_HEAD_UNITS, data_augmentation
    )
    
    # Compile ViT model
    optimizer = tfa.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    
    vit_classifier.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    
    vit_classifier.summary()
    
    # Train ViT model
    print(f"\nTraining ViT model for {EPOCHS_VIT} epochs...")
    checkpoint_filepath = "./vit_checkpoint"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    
    vit_history = vit_classifier.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_VIT,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_callback],
    )
    
    # Load best weights
    vit_classifier.load_weights(checkpoint_filepath)
    
    # Evaluate ViT model
    print("\nEvaluating ViT model...")
    _, vit_accuracy, vit_top_5_accuracy = vit_classifier.evaluate(x_test, y_test)
    print(f"ViT Model Accuracy: {round(vit_accuracy * 100, 2)}%")
    print(f"ViT Model Top-5 Accuracy: {round(vit_top_5_accuracy * 100, 2)}%")
    
    # Plot ViT training history
    plot_training_history(vit_history, 'ViT Training History - Optimizer: AdamW',
                         'vit_training_history.png')
    
    # ViT predictions and confusion matrix
    print("\nGenerating ViT predictions...")
    vit_predictions = vit_classifier.predict(x_test)
    vit_y_pred = [np.argmax(probas) for probas in vit_predictions]
    
    vit_cnf_matrix = confusion_matrix(y_test, vit_y_pred)
    plot_confusion_matrix(vit_cnf_matrix, classes=["Yes", "No"], 
                         title='ViT Normalized Confusion Matrix',
                         save_path='vit_confusion_matrix.png')
    
    print("\nViT Classification Report:")
    print(classification_report(y_test, vit_y_pred, target_names=["No Tumor", "Tumor"]))
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"CNN Model Accuracy: {round(cnn_results[1]*100, 2)}%")
    print(f"ViT Model Accuracy: {round(vit_accuracy * 100, 2)}%")
    print(f"ViT Model Top-5 Accuracy: {round(vit_top_5_accuracy * 100, 2)}%")
    print("\nTraining complete! Check the saved plots for detailed results.")
    print("=" * 80)

if __name__ == "__main__":
    main()