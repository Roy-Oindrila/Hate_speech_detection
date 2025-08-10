🛡️ Hate Speech Detection App


This repository contains a **machine learning model** and a **Streamlit web application** for detecting hate speech in text, specifically tweets.  
The model is trained on the **Twitter Sentiment Analysis Dataset** from Kaggle, and the web app allows users to input text and classify it as either **Hate Speech** or **Non-Hate Speech**.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Files in Repository](#files-in-repository)
- [Run in Google Colab](#run-in-google-colab)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Deploying the Streamlit App](#deploying-the-streamlit-app)
- [License](#license)

---

## 📖 Project Overview
This project has **two main components**:

1. **Model Training** – Implemented in `Hate_Speech_Detection.ipynb`:
   - Downloads the dataset
   - Cleans and preprocesses text
   - Trains a **Logistic Regression** classifier using **TF-IDF vectorization**
   - Saves the trained model and vectorizer for later use

2. **Web Application** – Implemented in `app.py`:
   - Loads the trained model and vectorizer
   - Classifies user-input text in real time
   - Displays predictions with confidence scores

---

## ✨ Features
✅ **Text Preprocessing** – Removes URLs, mentions, hashtags, punctuation, and numbers  
✅ **Machine Learning Model** – Logistic Regression with TF-IDF features  
✅ **Interactive Web Interface** – Built using Streamlit for live predictions  
✅ **Customizable Threshold** – Adjust prediction strictness  
✅ **Error Handling** – Handles missing files, wrong inputs, and runtime errors  

---

## 📊 Dataset
- **Name:** Twitter Sentiment Analysis Dataset (Kaggle)  
- **Labels:**
  - `0` → Hate Speech
  - `1` → Non-Hate Speech  
- Downloaded automatically using the Kaggle API during training.

---

🏋️‍♂️ Training the Model – How It Works
Download dataset from Kaggle

Clean text (remove URLs, hashtags, mentions, punctuation, numbers)

Split data into train/test sets

Vectorize text using TF-IDF

Train Logistic Regression model with balanced weights

Evaluate performance using confusion matrix & classification report

Save model & vectorizer as .pkl files

## 📂 Files in Repository

| File Name                  | Description |
|----------------------------|-------------|
| `Hate_Speech_Detection.ipynb`        | Colab/Jupyter notebook for training |
| `app.py`                   | Streamlit app for live classification |
| `hate_speech_model.pkl`    | Trained Logistic Regression model |
| `tfidf_vectorizer.pkl`     | Trained TF-IDF vectorizer |
| `requirements.txt`         | Python dependencies |
| `README.md`                | Project documentation |

---

## 🚀 Run in Google Colab
Click the badge below to open the training notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/your-username/hate-speech-detection/blob/main/train_model.ipynb](https://colab.research.google.com/drive/1FhARR77RaGfU-UAqxRw04tdyJvH1Su2T))

---

## 💻 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/hate-speech-detection.git
cd hate-speech-detection
pip install -r requirements.txt

🌐 Deploying the Streamlit App
To deploy:

Push your repository to GitHub

Connect to Streamlit Community Cloud

Set app.py as the entry point

Include both .pkl files in your repo



## Limitations
- **Prediction Accuracy Variability** – While many predictions are correct, some fail due to:
  - Ambiguous or sarcastic language
  - Slang and spelling variations
  - Cultural references
- **Data Bias** – Dataset may not represent all linguistic contexts.
- **Generalization Issues** – Lower accuracy on very different data.

