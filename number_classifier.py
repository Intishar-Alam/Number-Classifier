import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# ── Page setup ──────────────────────────────────────────────
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✏️ Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) on the pad below, then click **Predict**.")

# ── Load the trained models (cached so they only load once) ──
@st.cache_resource
def get_cnn_model():
    return load_model("mnist_cnn_model.h5")

@st.cache_resource
def get_ann_model():
    return load_model("ann_model.h5")

@st.cache_resource
def get_perceptron_model():
    return load_model("perceptron_model.h5")

model = get_cnn_model()
ann_model = get_ann_model()
perceptron_model = get_perceptron_model()

# ── Drawing canvas ────────────────────────────────────────────
# White strokes on a black background — matches MNIST's format
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True, 
)

predict_clicked = st.button("Predict", type="primary")

# ── Preprocessing + prediction ───────────────────────────────
# MNIST digits were not just resized — each one was cropped to its
# bounding box, scaled to fit a 20x20 box, then centered (by center
# of mass) inside a 28x28 frame. Matching that pipeline here makes a
# huge difference in accuracy for freehand-drawn digits.
def preprocess(image_data):
    # image_data is RGBA, shape (280, 280, 4)
    img = Image.fromarray((image_data[:, :, :3]).astype("uint8")).convert("L")
    arr = np.array(img).astype("float32")

    # 1. Crop tightly to the drawn strokes (bounding box of non-zero pixels)
    coords = np.argwhere(arr > 20)  # small threshold to ignore near-black noise
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = arr[y0:y1, x0:x1]

    # 2. Resize the cropped digit to fit inside a 20x20 box, preserving aspect ratio
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    cropped_img = Image.fromarray(cropped.astype("uint8")).resize(
        (new_w, new_h), Image.LANCZOS
    )
    resized = np.array(cropped_img).astype("float32")

    # 3. Paste onto a blank 28x28 canvas, centered
    canvas28 = np.zeros((28, 28), dtype="float32")
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas28[top:top + new_h, left:left + new_w] = resized

    # 4. Re-center based on center of mass, like the original MNIST pipeline
    total = canvas28.sum()
    if total > 0:
        ys, xs = np.indices(canvas28.shape)
        cy = (ys * canvas28).sum() / total
        cx = (xs * canvas28).sum() / total
        shift_y = int(round(14 - cy))
        shift_x = int(round(14 - cx))
        canvas28 = np.roll(canvas28, shift_y, axis=0)
        canvas28 = np.roll(canvas28, shift_x, axis=1)

    arr = canvas28 / 255.0
    arr = arr.reshape(1, 28, 28, 1)  # match CNN input shape
    return arr

if predict_clicked:
    if canvas_result.image_data is None or canvas_result.image_data[:, :, :3].sum() == 0:
        st.warning("Please draw a digit first.")
    else:
        processed = preprocess(canvas_result.image_data)  # shape (1, 28, 28, 1) — unchanged

        # Perceptron and ANN expect (1, 28, 28) — same pixel data, just reshaped
        flat_input = processed.reshape(1, 28, 28)

        perceptron_probs = perceptron_model.predict(flat_input, verbose=0)[0]
        ann_probs = ann_model.predict(flat_input, verbose=0)[0]
        cnn_probs = model.predict(processed, verbose=0)[0]

        st.markdown("## Model Comparison")
        col1, col2, col3 = st.columns(3)

        for col, name, probs in [
            (col1, "Perceptron", perceptron_probs),
            (col2, "ANN", ann_probs),
            (col3, "CNN", cnn_probs),
        ]:
            digit = int(np.argmax(probs))
            confidence = float(np.max(probs)) * 100
            with col:
                st.markdown(f"### {name}")
                st.markdown(f"**Prediction: {digit}**")
                st.markdown(f"Confidence: {confidence:.2f}%")
                st.bar_chart({str(i): float(probs[i]) for i in range(10)})