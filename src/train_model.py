# : Setup and Imports

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from google.colab import drive
# data set size is 10 images for defective class 50 images for non_defective class
# --- Configuration ---
# Image dimensions expected by the model
IMG_SIZE = 224
# Number of color channels (3 for RGB)
IMG_CHANNELS = 3
# Number of output classes (defective vs. non-defective)
NUM_CLASSES = 1
# Number of folds for cross-validation. 5 is a good standard for this dataset size.
N_SPLITS = 5
# Batch size for training
BATCH_SIZE = 16
# Number of epochs to train the head
HEAD_EPOCHS = 15
# Number of epochs to fine-tune the full model
FINETUNE_EPOCHS = 10
# Learning rate for the head training phase
HEAD_LR = 1e-3
# A very low learning rate for the fine-tuning phase
FINETUNE_LR = 1e-5

#  Connect to Google Drive and Define Paths

# This will prompt you for authorization.
print("Connecting to Google Drive...")
drive.mount('/content/drive')

# --- Define Paths ---

# I have  assumed this path you can have your own 
BASE_PATH = '/content/drive/My Drive/fabric_dataset/'
DATA_PATH = os.path.join(BASE_PATH, 'train')

# Check if the path exists to avoid errors
if not os.path.exists(DATA_PATH):
    print("ERROR: The specified data path does not exist.")
    print(f"Please make sure this path is correct: {DATA_PATH}")
else:
    print(f" executed: Google Drive mounted.")
    print(f"Data path is set to: {DATA_PATH}")

#  Load Image Filepaths and Labels

def load_filepaths_and_labels(data_path):
    """Loads image file paths and their corresponding labels."""
    filepaths = []
    labels = []
    # Ensure a consistent order ('defective', 'non_defective')
    class_names = sorted(os.listdir(data_path), reverse=True)

    print(f"Class names found: {class_names}")
    # We want 'defective' to be label 1 (the positive class)
    label_map = {'defective': 1, 'non_defective': 0}
    print(f"Label mapping: {label_map}")

    for class_name in class_names:
        class_dir = os.path.join(data_path, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                filepaths.append(filepath)
                labels.append(label_map[class_name])

    return np.array(filepaths), np.array(labels)

print("\nLoading filepaths and labels...")
filepaths, labels = load_filepaths_and_labels(DATA_PATH)

print(f"\nFound {len(filepaths)} images belonging to 2 classes.")
print(f"Total defective images: {np.sum(labels == 1)}")
print(f"Total non-defective images: {np.sum(labels == 0)}")
print(" executed: Filepaths and labels are loaded into memory.")

#  Define Data Preprocessing and Augmentation

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ],
    name="data_augmentation",
)

def preprocess_image(filepath, label):
    """Loads and preprocesses a single image."""
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=IMG_CHANNELS)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label

def create_dataset(X, y, is_training=True):
    """Creates a tf.data.Dataset from filepaths and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(X))
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(BATCH_SIZE)

    return dataset.prefetch(tf.data.AUTOTUNE)

print(" executed: Preprocessing and augmentation functions are defined.")

# Define the Model Building Function

def build_model(learning_rate=1e-3):
    """Builds the EfficientNetB0 model with a custom classifier head."""
    # Load the base model with pre-trained ImageNet weights
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    )

    # Freeze the base model layers initially
    base_model.trainable = False

    # Create the new model head
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

# Build a temporary model just to print the summary
print("Model architecture defined. Displaying summary:")
model_summary = build_model()
model_summary.summary()
print(" executed: Model building function is ready.")

#  Run Cross-Validation Training

print("Starting Stratified K-Fold Cross-Validation...")
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_histories = []
fold_no = 1

for train_index, val_index in kfold.split(filepaths, labels):
    print("-" * 60)
    print(f"Training Fold {fold_no}/{N_SPLITS}...")

    # Split data and create datasets
    X_train, X_val = filepaths[train_index], filepaths[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
    train_ds = create_dataset(X_train, y_train, is_training=True)
    val_ds = create_dataset(X_val, y_val, is_training=False)

    # Calculate class weights for the current fold's training data
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class weights for this fold: {class_weights_dict}")

    # Build and compile a fresh model for this fold
    model = build_model(learning_rate=HEAD_LR)

    # --- Phase 1: Train the Head ---
    print("\n--- Phase 1: Training the classifier head ---")
    history_head = model.fit(
        train_ds, epochs=HEAD_EPOCHS, validation_data=val_ds,
        class_weight=class_weights_dict, verbose=2
    )

    # --- Phase 2: Fine-Tuning ---
    print("\n--- Phase 2: Fine-tuning the full model ---")
    model.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    history_finetune = model.fit(
        train_ds, epochs=FINETUNE_EPOCHS, validation_data=val_ds,
        class_weight=class_weights_dict, verbose=2
    )

    fold_histories.append(history_finetune)
    fold_no += 1

print("\n executed: Cross-validation finished.")

#  Train and Save the Final Model

print("-" * 60)
print("Training the final model on the ENTIRE dataset...")

# Create the final dataset using all 60 images
final_train_ds = create_dataset(filepaths, labels, is_training=True)

# Calculate final class weights for all data
final_class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(labels), y=labels
)
final_class_weights_dict = dict(enumerate(final_class_weights))
print(f"Final model class weights: {final_class_weights_dict}")

# Build the final model
final_model = build_model(learning_rate=HEAD_LR)

# --- Final Model: Phase 1 (Head Training) ---
print("\n--- Final Model: Training Head ---")
final_model.fit(
    final_train_ds, epochs=HEAD_EPOCHS,
    class_weight=final_class_weights_dict, verbose=2
)

# --- Final Model: Phase 2 (Fine-Tuning) ---
print("\n--- Final Model: Fine-Tuning ---")
final_model.trainable = True
final_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
final_model.fit(
    final_train_ds, epochs=FINETUNE_EPOCHS,
    class_weight=final_class_weights_dict, verbose=2
)

# --- Save the Final Model ---
SAVE_PATH = os.path.join(BASE_PATH, 'final_fabric_defect_model.keras')
final_model.save(SAVE_PATH)

print("-" * 60)
print(f"Final model successfully trained and saved to your Google Drive!")
print(f"File location: {SAVE_PATH}")
