import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from warnings import filterwarnings
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, confusion_matrix as sk_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


filterwarnings('ignore')


BASE_DATA_DIR = 'leapGestRecog' 

IMG_SIZE = 100 
TEST_SIZE_RATIO = 0.2

# --- Dataset Subset Configuration (Crucial for managing processing time) ---
# Set the number of images to load per category (00, 01...09) for the ENTIRE datase
# Reduce these numbers significantly if processing is too slow.
LIMIT_PER_CATEGORY_TOTAL = 500 # Load up to 500 images per class (total 5000 images if 10 classes)

# Define the categories and their labels
# These will be dynamically detected as folder names, but explicit definition helps for clarity/ordering
CATEGORIES_LIST = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09'] # Expecting 10 classes
LABELS_DICT = {cls_name: i for i, cls_name in enumerate(CATEGORIES_LIST)}


print("--- Starting Hand Gesture Recognition Project (Task 04) ---")
print(f"TensorFlow version: {tf.__version__}")

# --- 1. Verifying Data Directory Structure & Identifying Classes ---
print("\n--- 1. Verifying Data Directory Structure & Identifying Classes ---")

if not os.path.exists(BASE_DATA_DIR):
    print(f"Error: Base data directory '{BASE_DATA_DIR}' not found. This should not happen after 'dir' check.")
    print(f"Please ensure '{BASE_DATA_DIR}' folder is directly inside your Project_4 folder.")
    exit()

# Get the list of all gesture classes (subfolder names like '00', '01'...)
# This will also detect 'leapGestRecog' if it's a subfolder in itself (which it shouldn't be)
gesture_classes_detected = sorted([name for name in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, name))])

# Filter out any unexpected folders if the dataset structure includes its own name as a folder
# This line is added to filter out 'leapGestRecog' if it's mistakenly detected as a class
gesture_classes = [cls for cls in gesture_classes_detected if cls in CATEGORIES_LIST]


if not gesture_classes:
    print(f"Error: No valid gesture class folders (00-09) found inside '{BASE_DATA_DIR}'.")
    print("Please ensure your dataset contains subfolders for each gesture (e.g., 'leapGestRecog/00/', 'leapGestRecog/01/').")
    exit()

print(f"Detected {len(gesture_classes)} valid gesture classes: {gesture_classes}")

# --- 2. Load Images and Labels ---
print("\n--- 2. Loading Images and Labels ---")
print(f"Loading up to {LIMIT_PER_CATEGORY_TOTAL} images per class for total dataset...")
print("This may take a while depending on dataset size and limits...")

def load_and_preprocess_images(base_dir, gesture_classes_list, img_size, limit_per_category=None):
    data = []
    skipped_corrupt_images = 0
    class_counts = {cls: 0 for cls in gesture_classes_list} # Initialize counts for valid classes only

    for class_name in tqdm(gesture_classes_list, desc="Processing Classes"): # Iterate through the *valid* classes
        class_path = os.path.join(base_dir, class_name)
        label = CATEGORIES_LIST.index(class_name) # Assign numerical label based on ordered CATEGORIES_LIST

        # --- UPDATED GLOB PATTERNS HERE ---
        # Try finding images one level deeper (e.g., 00/01_palm/*.jpg)
        image_files_in_class = []
        image_files_in_class.extend(glob.glob(os.path.join(class_path, '*', '*.jpg'))) # Pattern: class_folder/subfolder/*.jpg
        image_files_in_class.extend(glob.glob(os.path.join(class_path, '*', '*.png'))) # Pattern: class_folder/subfolder/*.png

        # If still nothing, try another level deeper (e.g., 00/01_palm/user_id/*.jpg)
        if not image_files_in_class:
            image_files_in_class.extend(glob.glob(os.path.join(class_path, '*', '*', '*.jpg'))) # Pattern: class_folder/sub-subfolder/user_id/*.jpg
            image_files_in_class.extend(glob.glob(os.path.join(class_path, '*', '*', '*.png'))) # Pattern: class_folder/sub-subfolder/user_id/*.png

        # If still nothing, try looking directly in the class folder itself (less common for this dataset)
        if not image_files_in_class:
            image_files_in_class.extend(glob.glob(os.path.join(class_path, '*.jpg'))) # Pattern: class_folder/*.jpg
            image_files_in_class.extend(glob.glob(os.path.join(class_path, '*.png'))) # Pattern: class_folder/*.png

        if not image_files_in_class:
            tqdm.write(f"Warning: No images found for class {class_name} at expected depths. Skipping.")
            continue # Skip to next class if no images found

        for img_file in image_files_in_class:
            try:
                if limit_per_category is not None and class_counts[class_name] >= limit_per_category:
                    break # Break inner loop if limit for this class is reached

                img_array = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    raise IOError("Image failed to load")

                img_resized = cv2.resize(img_array, (img_size, img_size))
                images_normalized = img_resized / 255.0
                data.append([images_normalized, label])
                class_counts[class_name] += 1
            except Exception as e:
                skipped_corrupt_images += 1

    return data, skipped_corrupt_images, class_counts

all_data, skipped_total, counts_per_class = load_and_preprocess_images(BASE_DATA_DIR, gesture_classes, IMG_SIZE, LIMIT_PER_CATEGORY_TOTAL)

print(f"\nSuccessfully loaded {len(all_data)} images across all classes (skipped {skipped_total} corrupt).")
print("Image counts per class (after limit):")
for cls, count in counts_per_class.items():
    print(f"  Class {cls}: {count} images")

# Convert to NumPy arrays
X = np.array([item[0] for item in all_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array([item[1] for item in all_data])

print(f"\nImages array shape (X): {X.shape} (Number of images, Height, Width, Channels)")
print(f"Labels array shape (y): {y.shape}")

print("\n--- Image Loading and Initial Preprocessing Complete ---")

# --- 3. Feature Extraction (HOG) ---
# HOG features are not typically used directly with CNNs. CNNs learn features automatically.
# This section is commented out for a standard CNN approach.

# --- 4. Prepare Data for CNN Training ---
print("\n--- 4. Preparing Data for CNN Training (Train/Test Split) ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE_RATIO, random_state=42, stratify=y
)

print(f"Training set features shape: {X_train.shape}")
print(f"Training set labels shape: {y_train.shape}")
print(f"Test set features shape: {X_test.shape}")
print(f"Test set labels shape: {y_test.shape}")

print("\n--- Data Preparation for CNN Complete ---")


# --- 5. Build and Train CNN Model (Deep Learning) ---
print("\n--- 5. Building and Training CNN Model (Deep Learning) ---")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train, num_classes=len(gesture_classes))
y_test_one_hot = to_categorical(y_test, num_classes=len(gesture_classes))

print(f"Train labels (one-hot) shape: {y_train_one_hot.shape}")
print(f"Test labels (one-hot) shape: {y_test_one_hot.shape}")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(gesture_classes), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\nTraining CNN model... (This will take significant time and show training progress)")
history = model.fit(X_train, y_train_one_hot,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=1)

print("\n--- CNN Model Training Complete ---")


# --- 6. Model Evaluation ---
print("\n--- 6. Model Evaluation ---")

print("Evaluating model on test set...")
loss, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=gesture_classes))

cm = sk_confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=gesture_classes, yticklabels=gesture_classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Hand Gesture Recognition')
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n--- Model Evaluation Complete ---")
print("Project finished!")