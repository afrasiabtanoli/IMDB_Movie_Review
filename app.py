import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("lstm_review_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LENGTH = 200   # SAME as your model input_length

st.title("Movie Review Predictor")

review = st.text_area("Enter your review here:")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([review])

        # Pad sequence
        padded = pad_sequences(sequence, maxlen=MAX_LENGTH)

        # Predict
        prediction = model.predict(padded)

        probability = prediction[0][0]

        percentage = probability * 100

        st.write(f"Prediction Probability: {percentage:.2f}%")

        if probability > 0.5:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")
