# ✏️ Handwritten Digit Recognizer — Perceptron vs ANN vs CNN

My first deep learning project — a handwritten digit classifier trained on the MNIST
dataset using three different architectures (Perceptron, ANN, and CNN), with a live
Streamlit app that lets you draw a digit and see how each model predicts it,
side by side.

## 🎯 Project Overview

This project explores how model architecture affects performance on the same
image classification task. Rather than building just one model, I trained three
increasingly complex architectures on identical data and compared their real-world
performance — not just on the test set, but on **freehand digits I draw myself**
through an interactive web app.

The goal: see, visually and numerically, why a simple Perceptron struggles where
a CNN thrives — using the exact same drawn input for all three models at once.

🔗 Live Demo: number-classifier-project.streamlit.app

## 📊 Dataset

- **Dataset:** MNIST handwritten digits (0–9)
- **Image size:** 28×28 grayscale pixels
- **Split:** standard train/test split, with validation performed on the test set
  during training (`validation_data=(X_test, y_test_cat)`)
- **Labels:** one-hot encoded via `to_categorical()` for 10 classes

## 🧠 Models & Results

All three models were trained for **5 epochs**, batch size **32**, using
`categorical_crossentropy` loss.

| Model          | Architecture Summary                                                                 | Optimizer | Test Accuracy |
|----------------|----------------------------------------------------------------------------------------|-----------|:--------------:|
| **Perceptron** | `Flatten → Dense(10, softmax)`                                                          | SGD       | **90.91%**     |
| **ANN**        | `Flatten → Dense(128, relu) → Dense(64, relu) → Dense(10, softmax)`                     | Adam      | **97.43%**     |
| **CNN**        | `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128,64,32, relu) → Dropout(0.5) → Dense(10, softmax)` | Adam      | **99.21%**     |

### Key takeaway
Accuracy climbs sharply with architectural complexity:
- The **Perceptron** (a single linear layer) can only draw straight decision
  boundaries in pixel space — it gets the "easy" digits right but struggles with
  more ambiguous handwriting.
- The **ANN** adds non-linear hidden layers, letting it learn much more complex
  patterns, closing most of the gap.
- The **CNN** adds convolution and pooling layers that understand *spatial*
  structure (edges, curves, strokes) rather than treating each pixel independently —
  giving it the best generalization, especially on messy, freehand-drawn digits
  that don't look exactly like the clean MNIST training data.

This gap becomes very visible in the live app: draw an oddly-shaped or slanted
digit, and it's common to see the Perceptron and ANN disagree or misclassify it
while the CNN gets it right.

## 🖥️ Live Demo — Model Comparison App

The Streamlit app lets you:
1. **Draw a digit (0–9)** on an interactive canvas using your trackpad, mouse, or touchscreen
2. Click **Predict**
3. See **all three models' predictions side by side** — each with its predicted
   digit, confidence percentage, and full probability breakdown across all 10 digits

This makes it easy to directly compare *where* and *why* the simpler models fail
compared to the CNN, using the exact same input.

### Preprocessing pipeline

Freehand-drawn digits don't naturally match MNIST's format, so the app replicates
MNIST's actual preprocessing steps rather than just resizing the canvas:

1. **Crop** the drawing tightly to its bounding box (removing empty canvas space)
2. **Resize** the cropped digit to fit within a 20×20 box, preserving aspect ratio
3. **Center** it inside a 28×28 frame
4. **Re-center** based on center of mass (matching MNIST's original normalization),
   so the digit sits exactly where the models expect it

This step made a significant difference in real-world prediction accuracy —
without it, even a clean, correctly-drawn digit was frequently misclassified.

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow / Keras** — model building and training
- **Streamlit** — interactive web app
- **streamlit-drawable-canvas** — freehand drawing input
- **NumPy / Pillow** — image preprocessing
- **Google Colab** — model training environment
- **Streamlit Community Cloud** — deployment

## 📁 Project Structure

```
Number_classifier/
├── number_classifier.py     # Streamlit app (canvas, preprocessing, model comparison)
├── requirements.txt         # Dependencies for local + cloud deployment
├── mnist_cnn_model.h5        # Trained CNN model
├── ann_model.h5              # Trained ANN model
├── perceptron_model.h5        # Trained Perceptron model
├── CNN.ipynb                 # Colab notebook: perceptron, ANN, and CNN training
└── README.md
```

## 🚀 Running Locally

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd Number_classifier
```

**2. Create and activate a virtual environment** (Python 3.11 recommended)
```bash
python3.11 -m venv .env
source .env/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run number_classifier.py
```

The app will open automatically at `http://localhost:8501`.

## ☁️ Deployment

This app is deployed on **Streamlit Community Cloud**. Key deployment notes:

- `requirements.txt` must be committed to the repo root — the cloud build installs
  packages from it fresh, so anything installed only in a local environment won't
  carry over.
- All three `.h5` model files must be committed to the repo so the app can load them
  at runtime. Files over GitHub's 100MB limit require **Git LFS**.

## 📈 Possible Future Improvements

- Add **data augmentation** (rotation, shifting, slight distortion) during training
  to improve robustness against messy freehand input
- Add a **confusion matrix** view per model to see exactly which digits each
  architecture struggles with
- Experiment with deeper CNN architectures or batch normalization for further
  accuracy gains
- Allow uploading a custom test image in addition to drawing

## AI USage
- Taken Help from CLaude Code to design and code the UI in Streamlit

## 👤 Author

**Md Intishar Alam**
First deep learning project — built while learning Perceptron, ANN, and CNN
fundamentals through hands-on experimentation with the MNIST dataset.
