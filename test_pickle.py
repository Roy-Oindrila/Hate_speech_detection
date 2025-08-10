import pickle

# Test loading model
try:
    with open('hate_speech_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Test loading vectorizer
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded successfully")
    # Check if vectorizer is fitted
    if hasattr(vectorizer, 'idf_'):
        print("Vectorizer is fitted (has idf_ attribute)")
    else:
        print("Vectorizer is NOT fitted (missing idf_ attribute)")
except Exception as e:
    print(f"Error loading vectorizer: {e}")