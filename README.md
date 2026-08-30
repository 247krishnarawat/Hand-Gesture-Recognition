# Hand Gesture Recognition using CNN & Classical Computer Vision

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=flat-square&logo=keras&logoColor=white" alt="Keras" />
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Scikit_Learn-Machine%20Learning-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-Learn" />
</p>

---

## 📖 Overview

This project implements an end-to-end Computer Vision and Deep Learning pipeline to classify human hand gestures into 10 distinct gesture categories using the **LeapGestRecog Dataset** from Kaggle.

The project evaluates both a modern **Convolutional Neural Network (CNN)** built with TensorFlow/Keras and a classical Machine Learning pipeline combining **Histogram of Oriented Gradients (HOG)** feature extraction with an optimized **Support Vector Machine (SVM)** classifier.

---

## 🖐️ Gesture Classes

The dataset comprises grayscale infrared hand gesture captures across 10 defined categories:
1. `01_palm` — Flat palm facing camera
2. `02_l` — L-shape gesture
3. `03_fist` — Closed fist
4. `04_fist_moved` — Fist with movement trajectory
5. `05_thumb` — Extended thumb gesture
6. `06_index` — Pointing index finger
7. `07_ok` — OK hand sign
8. `08_palm_moved` — Palm motion
9. `09_c` — C-shape hand posture
10. `10_down` — Downward pointing hand

---

## 🔬 Model Architectures & Pipeline

### 1. Convolutional Neural Network (CNN)
- **Input Preprocessing**: Grayscale image normalization, resizing to uniform dimensions ($100 \times 100 \times 1$).
- **Feature Extraction**: Cascaded `Conv2D` layers with ReLU activation followed by `MaxPooling2D` and `Dropout` regularization to mitigate overfitting.
- **Classification Head**: `Flatten` layer into `Dense` layers with `Softmax` output for 10-class multi-class probability distribution.
- **Optimization**: Trained using the `Adam` optimizer with `Categorical Cross-Entropy` loss.

### 2. Classical Pipeline (HOG + SVM)
- **Feature Descriptor**: `scikit-image` HOG extraction (orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)).
- **Feature Scaling**: `StandardScaler` normalization of extracted gradient vectors.
- **Classifier**: Non-linear `SVC` with Radial Basis Function (`rbf`) kernel.
- **Hyperparameter Optimization**: `GridSearchCV` 3-fold cross-validation tuning penalty parameter `C` and kernel coefficient `gamma`.

---

## 📊 Evaluation & Metrics

The models are thoroughly evaluated on an independent test split ($20\%$ test ratio):
- **Accuracy Score**: Overall classification accuracy.
- **Classification Report**: Precision, Recall, and F1-Score for each of the 10 gesture classes.
- **Confusion Matrix**: Seaborn heatmap highlighting class-specific true positives and misclassifications.

---

## 📁 Repository Structure

```text
Hand-Gesture-Recognition/
├── gesture_recognizer.py    # Complete training, evaluation & inference script
├── leapGestRecog/           # Dataset directory (LeapGestRecog images)
├── README.md                # Project documentation
└── requirements.txt         # Project dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Recommended: GPU environment (CUDA) for accelerated CNN training

---

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/247krishnarawat/Hand-Gesture-Recognition.git
   cd Hand-Gesture-Recognition
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install opencv-python tensorflow scikit-learn scikit-image matplotlib seaborn tqdm
   ```

4. **Download the Dataset:**
   - Download the [LeapGestRecog Dataset from Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).
   - Extract the dataset into the project root under the `leapGestRecog/` directory.

5. **Run Training & Evaluation:**
   ```bash
   python gesture_recognizer.py
   ```

---

## 🔮 Future Improvements

- 📹 Real-time webcam gesture recognition using OpenCV video stream capture
- 📱 Lightweight model export using TensorFlow Lite (`.tflite`) for edge deployment
- 🖐️ MediaPipe hand landmark integration for real-time 3D coordinate tracking
