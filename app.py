import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Indian Grain Classifier",
    page_icon="🌾",
    layout="centered"
)

# Class names (alphabetical order, matches training)
CLASS_NAMES = ['chaana_dal', 'chole', 'harbara', 'masur_dal', 'matki',
               'moong', 'peanut', 'rice', 'tur_dal', 'wheat']

DISPLAY_NAMES = {
    'chaana_dal': 'Chana Dal',
    'chole': 'Chole (Chickpeas)',
    'harbara': 'Harbara',
    'masur_dal': 'Masur Dal',
    'matki': 'Matki (Moth Beans)',
    'moong': 'Moong',
    'peanut': 'Peanut',
    'rice': 'Rice',
    'tur_dal': 'Tur Dal',
    'wheat': 'Wheat'
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("grain_classifier.keras")

model = load_model()

def predict(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx]

    return predicted_class, confidence, predictions

# --- UI ---
st.title("🌾 Indian Grain Classifier")
st.markdown(
    "Upload an image of a grain, dal, or legume to classify it into one of "
    "**10 categories**: Chana Dal, Chole, Harbara, Masur Dal, Matki, Moong, "
    "Peanut, Rice, Tur Dal, or Wheat."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        predicted_class, confidence, all_predictions = predict(image)

    with col2:
        st.subheader("Prediction")
        st.markdown(f"### {DISPLAY_NAMES[predicted_class]}")
        st.metric("Confidence", f"{confidence*100:.2f}%")

        if confidence < 0.5:
            st.warning("Low confidence — the image may be unclear or an unfamiliar variety.")

    st.markdown("---")
    st.subheader("Confidence across all classes")

    # Sort by confidence descending for display
    sorted_indices = np.argsort(all_predictions)[::-1]
    for idx in sorted_indices:
        class_name = CLASS_NAMES[idx]
        prob = all_predictions[idx]
        st.progress(float(prob), text=f"{DISPLAY_NAMES[class_name]}: {prob*100:.2f}%")

else:
    st.info("👆 Upload an image to get started.")

st.markdown("---")
st.caption(
    "Model: EfficientNetB3 (transfer learning) | "
    "Trained on 10 Indian grain/legume classes | "
    "Test accuracy: ~83%"
)