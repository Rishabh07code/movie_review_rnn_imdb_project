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
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Reverse mapping
index_word = {v: k for k, v in tokenizer.word_index.items()}

# Prediction function
def predict_next_word(text):
    if not text.strip():
        return "Please enter valid input"

    max_sequence_len = model.input_shape[1] + 1

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    predicted = model.predict(token_list, verbose=0)

    predicted_index = np.argmax(predicted, axis=1)[0]

    return index_word.get(predicted_index, "Word not found")


# UI
st.title("Next Word Prediction (LSTM)")

input_text = st.text_input("Enter text", "To be or not to")

if st.button("Predict"):
    result = predict_next_word(input_text)
    st.write(f"Next word: {result}")
