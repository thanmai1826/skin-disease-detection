import streamlit as st
from src.predict import predict_disease
import os

# Page Configuration
st.set_page_config(
    page_title="Skin Disease Detector",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.title("🩺 Skin Disease Detection System")
    st.markdown("### Using Convolutional Neural Networks (CNN)")
    st.write("Upload a skin image to detect: **Acne, Eczema, Psoriasis, Melanoma, or Normal**.")
    
    st.divider()

    # Image Upload
    uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        st.write("") # Spacing
        
        # Predict Button
        if st.button("Detect Disease"):
            with st.spinner("Analyzing image..."):
                try:
                    # Save the uploaded file temporarily to pass to predict function
                    # (Streamlit handles in-memory files, but our predict func expects a path usually. 
                    # Let's save it temporarily)
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp