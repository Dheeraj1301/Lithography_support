import sys
import os

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.controller.adaptive_controller import AdaptiveLithoController
import os

# Correct model path
model_path = "checkpoints/lstm_model.pth"

# Check if model file exists before loading
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please check the path.")
    st.stop()

# Initialize controller with correct model path
controller = AdaptiveLithoController(model_path)

st.title("Lithography AI Controller")

user_input = st.text_input("Enter comma-separated features (4 features expected)")

if st.button("Predict"):
    try:
        # Convert input string to list of floats
        features = [float(x.strip()) for x in user_input.split(",")]
        
        # Check input length matches model input_size (4)
        if len(features) != 4:
            st.error("Please enter exactly 4 features separated by commas.")
        else:
            prediction = controller.predict(features)
            st.success(f"Predicted CD: {prediction:.2f} nm")
    except Exception as e:
        st.error(f"Invalid input: {e}")
