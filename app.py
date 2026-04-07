import streamlit as st
import numpy as np
import pickle
import os


# Base directory
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "next_word_lstm_model_with_early_stopping.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pickle")

# Cache model (important)
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH, compile=False)



# Load tokenizer
try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    tokenizer = None

# Reverse mapping
index_word = {}

# Prediction function
def predict_next_word(text):
    return "Model temporarily disabled (TensorFlow not installed)"

# UI
st.title("Next Word Prediction (LSTM)")

input_text = st.text_input("Enter text", "To be or not to")

if st.button("Predict"):
    result = predict_next_word(input_text)
    st.write(f"Next word: {result}")
