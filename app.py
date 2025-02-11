
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import pickle
import nltk
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import os

# Configure NLTK data directory
def setup_nltk():
    nltk_data_dir = Path("./nltk_data")
    nltk_data_dir.mkdir(exist_ok=True)
    nltk.data.path.append(str(nltk_data_dir))
    
    resources = ["punkt", "stopwords", "wordnet"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, download_dir=str(nltk_data_dir))

# Safe pickle loading function
def safe_pickle_load(file_path):
    try:
        with open(file_path, 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

# Load model and resources safely
@st.cache_resource
def load_resources():
    try:
        # Load the model
        if not os.path.exists('sentimen_model.h5'):
            st.error("Model file 'sentimen_model.h5' not found!")
            return None, None, None, None
        
        model = keras.models.load_model('sentimen_model.h5')
        
        # Load other resources
        tokenizer = safe_pickle_load('tokenizer.pkl')
        label_encoder = safe_pickle_load('label_encoderA.pkl')
        maxlen = safe_pickle_load('maxlen.pkl')
        
        if None in (tokenizer, label_encoder, maxlen):
            return None, None, None, None
            
        return model, tokenizer, label_encoder, maxlen
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None, None, None

def preprocessing_text(text):
    """Preprocess the input text."""
    try:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z0-9\s\?!.,\'"]', '', text)
        
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        important_words = {'not', 'no', "n't"}
        words = [word for word in words if word not in stop_words or word in important_words]
        
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return text

def check_environment():
    """Check if the environment is properly set up"""
    try:
        import numpy
        import sklearn
        st.sidebar.success(f"NumPy version: {numpy.__version__}")
        st.sidebar.success(f"scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        st.error(f"Missing required dependencies: {str(e)}")
        st.stop()

def main():
    st.title('Klasifikasi Jenis Pertanyaan Menggunakan Machine Learning')
    
    # Check environment
    check_environment()
    
    # Setup NLTK
    setup_nltk()
    
    # Load resources
    model, tokenizer, label_encoder, maxlen = load_resources()
    
    if None in (model, tokenizer, label_encoder, maxlen):
        st.error("Failed to load required resources. Please check if all model files are present.")
        st.info("Required files: 'sentimen_model.h5', 'tokenizer.pkl', 'label_encoderA.pkl', 'maxlen.pkl'")
        return
    
    # Input text
    text = st.text_input("Masukkan Pertanyaan:", key="input1")
    
    if text.strip():
        try:
            # Preprocess text
            text_prepared = preprocessing_text(text)
            
            # Convert to sequence
            sequence_testing = tokenizer.texts_to_sequences([text_prepared])
            padded_testing = pad_sequences(sequence_testing, maxlen=maxlen, padding='post')
            
            # Make prediction
            with st.spinner('Making prediction...'):
                prediksi = model.predict(padded_testing)
                predicted_class = np.argmax(prediksi, axis=1)[0]
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            
            # Display results
            st.success("Prediction complete!")
            st.write("Hasil Prediksi (Class):", predicted_label)
            
            # Display confidence scores
            confidence = float(prediksi[0][predicted_class])
            st.write(f"Confidence Score: {confidence:.2%}")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please try again with a different input or contact support if the problem persists.")

if __name__ == "__main__":
    main()
