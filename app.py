import os
import keras
from keras.models import load_model
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =====================================================================
# 1. PAGE CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="Bloom Vision | Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS for grand look
custom_css = """
    <style>
        .stApp { background-color: #f0f2f6; }
        .title { font-family: 'Georgia', serif; color: #2c3e50; text-align: center; padding: 20px; }
        .result-card { background-color: #ffffff; border-radius: 15px; padding: 25px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); text-align: center; }
        .flower-name { font-size: 28px; font-weight: bold; color: #e84393; }
        .confidence-score { font-size: 20px; color: #34495e; }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# =====================================================================
# 2. DATA & MODEL LOADING
# =====================================================================

# List of flower names (lowercase to match dictionary keys)
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# ===============================================
# ---> INGA THAAN PUDHU CHANGE <---
# Dictionary with flower information
# ===============================================
flower_info = {
    'daisy': """
    ### 🌼 Daisy
    **Name Origin:** The name comes from "Day's Eye," because the flower opens during the day.  
    
    **Meaning:** Symbolizes purity, innocence, and new beginnings.  
    
    **Uniqueness:** Although it looks like a single flower, it’s actually a collection of hundreds of tiny flowers.  
""",
'dandelion': """
    ### 🌸 Dandelion
    **Name Origin:** From the French word "dent-de-lion" (lion’s tooth), referring to its sharp leaves.  
    
    **Meaning:** Represents resilience and survival. Also seen as a symbol of wishes coming true.  
    
    **Uses:** Has medicinal properties and is used in foods like salads.  
""",
'rose': """
    ### 🌹 Rose
    **Meaning:** Universally recognized as a symbol of love and beauty. A red rose represents love, while a yellow rose represents friendship.  
    
    **Uniqueness:** The thorns on a rose plant are not true "thorns" botanically—they are actually called "prickles."  
    
    **Uses:** Used in perfumes, cosmetics, and food items like rose candy.  
""",
'sunflower': """
    ### 🌻 Sunflower
    **Name Origin:** Named because of its trait of turning towards the sun (Heliotropism).  
    
    **Meaning:** Symbolizes adoration, loyalty, and longevity. Its brightness is seen as a symbol of happiness.  
    
    **Economic Value:** Its seeds are used to extract cooking oil and also eaten as a nutritious snack.  
""",
'tulip': """
    ### 🌷 Tulip
    **Origin:** Although often associated with the Netherlands, tulips actually originated in Central Asia.  
    
    **Meaning:** Symbolizes "perfect love" and the arrival of spring.  
    
    **Uniqueness:** During the 17th century "Tulip Mania," the price of a single tulip bulb was higher than the cost of a house!  
"""

}


# Load your trained model
try:
    model = load_model("Flower_Recog_Model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to classify the image
def classify_image(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    
    flower_name = flower_names[np.argmax(result)]
    confidence_score = np.max(result) * 100
    
    return flower_name, confidence_score

# =====================================================================
# 3. UI LAYOUT & INTERACTION
# =====================================================================

# Main Title
st.markdown('<h1 class="title">🌸 Bloom Vision: The AI Flower Identifier 🌸</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #555;">Upload an image of a flower, and our AI will tell you what it is!</p>', unsafe_allow_html=True)

# File uploader centered
_, col2_uploader, _ = st.columns([1, 2, 1])
with col2_uploader:
    uploaded_file = st.file_uploader(
        'Choose a flower image...', 
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    file_path = os.path.join("upload", uploaded_file.name)
    os.makedirs("upload", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Create two columns for image and result
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(file_path, caption='Your Uploaded Flower', use_column_width=True)

    with col2:
        with st.spinner('Our AI is smelling the flowers... 🧐'):
            flower_name, confidence_score = classify_image(file_path)
        
        st.success('Classification Complete!')

        st.markdown(
            f"""
            <div class="result-card">
                <h3>Prediction</h3>
                <p class="flower-name">{flower_name.capitalize()}</p>
                <p class="confidence-score">Confidence: <strong>{confidence_score:.2f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True) # Adding some space

        # ===============================================
        # ---> INGA THAAN INFORMATION-A KAATROM <---
        # Using st.expander to show flower info
        # ===============================================
        
        # Retrieve the info from our dictionary
        flower_details = flower_info.get(flower_name, "Sorry, information for this flower is not available.")

        with st.expander(f"👉 Click here to know more about {flower_name.capitalize()}"):
            st.markdown(flower_details, unsafe_allow_html=True)

        if confidence_score > 85:
            st.balloons()

# Sidebar
st.sidebar.title("About Bloom Vision")
st.sidebar.info(
    """
    This AI can classify five types of flowers:
    - Daisy 🌼
    - Dandelion 🌸
    - Rose 🌹
    - Sunflower 🌻
    - Tulip 🌷
    """
)