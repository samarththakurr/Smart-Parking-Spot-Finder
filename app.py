import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# 1. Set Page
st.set_page_config(page_title="🚗 Smart Parking Spot Finder", layout="centered")
st.title("🅿️ Vehicle Type Classifier and Parking Slot Checker")
st.markdown("Upload an image of an entering vehicle to predict its type and check parking availability.")

# 2. Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('vehicle_model.h5')
    return model

model = load_model()

# 3. Define Classes
classes = ['Bus', 'Car', 'Truck', 'motorcycle']

# 4. Set Slot Capacities
slot_capacity = {'Bike': 5, 'Car': 10, 'Truck': 2, 'Bus': 3}
current_vehicles = {'Bike': 3, 'Car': 7, 'Truck': 2, 'Bus': 1}

# 5. Upload image
uploaded_file = st.file_uploader("📤 Upload Vehicle Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Vehicle", use_column_width=True)

    # Preprocessing
    IMG_SIZE = 224
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions)
    predicted_class = classes[predicted_index]
    confidence = float(np.max(predictions))

    # ✅ Fix motorcycle to Bike
    if predicted_class == 'motorcycle':
        predicted_class = 'Bike'

    # Display prediction
    st.markdown("---")
    st.subheader("🧠 Prediction Result")
    st.success(f"**Vehicle Type:** `{predicted_class}`")
    st.info(f"**Confidence:** `{confidence:.2f}`")

    # Check Parking Availability
    if current_vehicles[predicted_class] < slot_capacity[predicted_class]:
        st.success(f"✅ Parking Available for {predicted_class}")
        st.balloons()
        remaining = slot_capacity[predicted_class] - current_vehicles[predicted_class]
        st.info(f"🅿️ Remaining slots for {predicted_class}: `{remaining}`")
        current_vehicles[predicted_class] += 1  # Update slot count after allowing entry
    else:
        st.error(f"❌ No Parking Available for {predicted_class}")
        st.warning("🚫 Barrier Stays Closed")

else:
    st.info("Please upload a vehicle image to classify and check parking!")
