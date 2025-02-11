

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

class QuestionClassifier:
    def __init__(self):
        self.setup_nltk()
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.maxlen = None
        
    def setup_nltk(self):
        """Setup NLTK resources"""
        nltk_data_dir = Path("./nltk_data")
        nltk_data_dir.mkdir(exist_ok=True)
        nltk.data.path.append(str(nltk_data_dir))
        
        for resource in ["punkt", "stopwords", "wordnet"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, download_dir=str(nltk_data_dir))

    def load_model(self):
        """Load all required model files"""
        try:
            self.model = keras.models.load_model('sentimen_model.h5')
            
            with open('tokenizer.pkl', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
                
            with open('label_encoder.pkl', 'rb') as handle:
                self.label_encoder = pickle.load(handle)
                
            with open('maxlen.pkl', 'rb') as handle:
                self.maxlen = pickle.load(handle)
                
            return True
        except FileNotFoundError as e:
            st.error(f"Missing file: {e.filename}")
            return False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

    def preprocess_text(self, text):
        """Preprocess input text"""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z0-9\s\?!.,\'"]', '', text)
        
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words or word in ['not', 'no', "n't"]]
        
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)

    def predict(self, text):
        """Make prediction on input text"""
        if not all([self.model, self.tokenizer, self.label_encoder, self.maxlen]):
            return None, None
            
        try:
            # Preprocess
            text_prepared = self.preprocess_text(text)
            
            # Tokenize and pad
            sequence = self.tokenizer.texts_to_sequences([text_prepared])
            padded = pad_sequences(sequence, maxlen=self.maxlen, padding='post')
            
            # Predict
            prediction = self.model.predict(padded)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            confidence = float(prediction[0][predicted_class])
            
            return predicted_label, confidence
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None, None

def main():
    st.title('Klasifikasi Jenis Pertanyaan Menggunakan Machine Learning')
    
    # Initialize classifier
    classifier = QuestionClassifier()
    
    # Load model
    with st.spinner('Loading model...'):
        if not classifier.load_model():
            st.error("Failed to load model. Please check if all required files are present.")
            st.stop()
    
    # Input
    text = st.text_input("Masukkan Pertanyaan:", key="input1")
    
    if text.strip():
        with st.spinner('Processing...'):
            label, confidence = classifier.predict(text)
            
            if label and confidence:
                st.success("Prediction complete!")
                st.write("Hasil Prediksi (Class):", label)
                st.write(f"Confidence Score: {confidence:.2%}")

if __name__ == "__main__":
    main()
