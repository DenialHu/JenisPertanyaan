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
    
    # Download required NLTK resources
    resources = ["punkt", "stopwords", "wordnet"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, download_dir=str(nltk_data_dir))

# Load model and resources safely
@st.cache_resource
def load_resources():
    try:
        # Load the model
        model = keras.models.load_model('sentimen_model.h5')
        
        # Load tokenizer
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # Load label encoder
        with open('label_encoderA.pkl', 'rb') as handle:
            label_encoder = pickle.load(handle)
            
        # Load maxlen
        with open('maxlen.pkl', 'rb') as handle:
            maxlen = pickle.load(handle)
            
        return model, tokenizer, label_encoder, maxlen
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None, None, None

def preprocessing_text(text):
    """Preprocess the input text."""
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\?!.,\'"]', '', text)
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords but keep negative words
        stop_words = set(stopwords.words('english'))
        important_words = {'not', 'no', "n't"}
        words = [word for word in words if word not in stop_words or word in important_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return text

def main():
    st.title('Klasifikasi Jenis Pertanyaan Menggunakan Machine Learning')
    
    # Setup NLTK
    setup_nltk()
    
    # Load resources
    model, tokenizer, label_encoder, maxlen = load_resources()
    
    if None in (model, tokenizer, label_encoder, maxlen):
        st.error("Failed to load required resources. Please check if all model files are present.")
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
            prediksi = model.predict(padded_testing)
            predicted_class = np.argmax(prediksi, axis=1)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            
            # Display results
            st.write("Hasil Prediksi (Class):", predicted_label)
            
            # Optional: Display confidence scores
            confidence = float(prediksi[0][predicted_class])
            st.write(f"Confidence Score: {confidence:.2%}")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
