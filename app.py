import streamlit as st
import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Hate Speech Detector", page_icon="🚫", layout="centered")

@st.cache_resource
def load_model_and_vectorizer():
    model_path = "hate_speech_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model or vectorizer files not found. Please ensure they are in the correct directory.")
        return None, None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        
        if not hasattr(vectorizer, 'idf_'):
            st.error("Loaded TfidfVectorizer is not fitted.")
            return None, None
            
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {str(e)}")
        return None, None

model, vectorizer = load_model_and_vectorizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

st.title("🚫 Hate Speech Detection App")
st.write("Enter a tweet or sentence to classify it as **Hate Speech** or **Non-Hate Speech**.")

user_input = st.text_area("Enter your text here:", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("Cannot analyze text because model or vectorizer failed to load.")
    else:
        try:
            cleaned = clean_text(user_input)
            vect_text = vectorizer.transform([cleaned])
            y_pred_proba = model.predict_proba(vect_text)[0]  # Get probabilities
            threshold = 0.3  # Adjust threshold (e.g., lower to 0.3 for more conservative hate speech detection)
            prediction = 1 if y_pred_proba[1] < threshold else 0  # 0 = Hate Speech, 1 = Non-Hate Speech

            labels = {
                0: "Hate Speech",
                1: "Non-Hate Speech"
            }

            st.subheader("Prediction:")
            st.success(f"🔍 The text is classified as: **{labels[prediction]}**")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")