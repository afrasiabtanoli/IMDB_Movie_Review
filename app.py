import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load Model & Tokenizer
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_lstm_model():
    model = load_model("lstm_review_model.keras")
    return model

@st.cache_resource(show_spinner=False)
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

model = load_lstm_model()
tokenizer = load_tokenizer()

# ------------------------------
# Constants
# ------------------------------
MAX_LENGTH = 200   # same as used in training

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(
    page_title="Movie Reviews Sentiment Analyzer",
    page_icon="📝",
    layout="centered"
)

st.title("Movie Reviews Sentiment Analyzer")
st.markdown(
    """
Enter a review below and click **Predict** to see if it's positive or negative along with probability.
"""
)

review_input = st.text_area("Enter your review here:", height=150)

if st.button("Predict"):

    if not review_input.strip():
        st.warning("Please enter a review first!")
    else:
        # Show spinner while predicting
        with st.spinner("Analyzing review..."):
            # Convert text to sequence
            seq = tokenizer.texts_to_sequences([review_input])
            padded = pad_sequences(seq, maxlen=MAX_LENGTH)

            # Predict
            pred = model.predict(padded, verbose=0)
            probability = pred[0][0]
            percentage = probability * 100

            # Display results
            if probability > 0.5:
                st.success(f"Positive Review 😊 ({percentage:.2f}%)")
                st.progress(min(int(percentage), 100))
            else:
                st.error(f"Negative Review 😞 ({100-percentage:.2f}%)")
                st.progress(min(int(100 - percentage), 100))

st.markdown("---")
st.markdown("Built By Afrasiab Tanoli with ❤️ using TensorFlow LSTM & Streamlit")
