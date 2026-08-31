import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="EcoVision AI",
    page_icon="♻️",
    layout="centered"
)

# -----------------------------
# PAGE DESIGN
# -----------------------------

st.title("♻️ EcoVision AI")
st.subheader("AI-Powered Waste Classification")

st.write(
    "Upload an image of waste and our AI model "
    "will classify it into one of six categories."
)

st.divider()

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "waste_classifier_mobilenetv2.keras"
    )

model = load_model()

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# -----------------------------
# UPLOAD IMAGE
# -----------------------------

uploaded_file = st.file_uploader(
    "📤 Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PREDICTION
# -----------------------------

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    if st.button("🔍 Classify Waste", type="primary"):

        with st.spinner("AI is analyzing the image..."):

            img = image.resize((224, 224))

            img_array = np.array(
                img,
                dtype=np.float32
            )

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                img_array
            )

            prediction = model.predict(
                img_array,
                verbose=0
            )[0]

            predicted_index = np.argmax(prediction)

            predicted_class = class_names[
                predicted_index
            ]

            confidence = (
                prediction[predicted_index] * 100
            )

        # -----------------------------
        # RESULT
        # -----------------------------

        st.success(
            f"♻️ Detected: {predicted_class.upper()}"
        )

        st.metric(
            "AI Confidence",
            f"{confidence:.2f}%"
        )

        # -----------------------------
        # ALL PROBABILITIES
        # -----------------------------

        st.subheader("📊 Prediction Probabilities")

        for i, class_name in enumerate(class_names):

            probability = float(prediction[i])

            st.write(
                f"**{class_name.capitalize()}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)

else:

    st.info(
        "👆 Upload a waste image to begin."
    )

st.divider()

st.caption(
    "EcoVision AI • TensorFlow • MobileNetV2 • Streamlit"
)
