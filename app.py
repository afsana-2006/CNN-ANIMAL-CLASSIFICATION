import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(page_title="Animal Detection AI", page_icon="🐾", layout="wide")

# Load the saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/model.h5')

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please run the specialized animal training script first.")
    model_loaded = False

# Specialized Animal Class Names
CLASS_NAMES = ['Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse']

# Title and Description
st.title("🐾 Specialized Animal Detection AI")
st.write("Upload an image of an animal to identify its species. (Supported: Bird, Cat, Deer, Dog, Frog, Horse)")

# File Uploader
uploaded_file = st.file_uploader("Choose an animal photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Animal Photo', use_container_width=True)
    
    # Preprocessing
    with st.spinner('Identifying Species...'):
        img = image.resize((32, 32))
        img_array = np.array(img)
        
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        elif len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
            
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = 100 * np.max(predictions[0])
        
    with col2:
        st.subheader("Analysis Result")
        st.success(f"Detected Species: **{CLASS_NAMES[class_idx]}**")
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
        
        # Probability bar chart
        st.write("### Probability Distribution")
        prob_dict = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(prob_dict)

st.markdown("---")
st.info("This model is trained specifically on common animal classes to provide higher accuracy for biological detection tasks.")
