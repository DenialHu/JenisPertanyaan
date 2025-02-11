import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import pickle
import nltk
import requests
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for loading status
if 'resources_loaded' not in st.session_state:
    st.session_state.resources_loaded = False

def setup_nltk():
    """Setup NLTK resources with proper error handling"""
    try:
        # Create nltk_data directory in the app's directory
        nltk_data_dir = Path("./nltk_data")
        nltk_data_dir.mkdir(exist_ok=True)
        nltk.data.path.append(str(nltk_data_dir))

        # Download required NLTK resources
        resources = ["punkt", "stopwords", "wordnet"]
        for resource in resources:
            try:
                nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
            except LookupError:
                with st.spinner(f'Downloading NLTK {resource}...'):
                    nltk.download(resource, download_dir=str(nltk_data_dir), quiet=True)
        return True
    except Exception as e:
        logger.error(f"Error setting up NLTK: {str(e)}")
        st.error(f"Failed to setup NLTK resources: {str(e)}")
        return False

def load_model_files():
    """Load all required model files with proper error handling"""
    try:
        # Load the keras model
        if not os.path.exists('sentimen_model.h5'):
            st.error("Model file 'sentimen_model.h5' not found!")
            return None
        
        model = keras.models.load_model('sentimen_model.h5')
        
        # Load tokenizer
        try:
            with open('tokenizer.pkl', 'rb') as handle:
                tokenizer = pickle.load(handle)
        except FileNotFoundError:
            st.error("Tokenizer file 'tokenizer.pkl' not found!")
            return None
        
        # Load label encoder from GitHub
        LABEL_ENCODER_URL = "https://github.com/DenialHu/JenisPertanyaan/raw/main/label_encoder.pkl"
        try:
            response = requests.get(LABEL_ENCODER_URL)
            if response.status_code == 200:
                label_encoder = pickle.loads(response.content)
            else:
                st.error(f"Failed to fetch label_encoder.pkl. Status code: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error loading label encoder: {str(e)}")
            return None
        
        # Load maxlen
        try:
            with open('maxlen.pkl', 'rb') as handle:
                maxlen = pickle.load(handle)
        except FileNotFoundError:
            st.error("Maxlen file 'maxlen.pkl' not found!")
            return None
            
        return model, tokenizer, label_encoder, maxlen
    except Exception as e:
        logger.error(f"Error loading model files: {str(e)}")
        st.error(f"Failed to load model files: {str(e)}")
        return None

def preprocess_text(text):
    """Preprocess input text with error handling"""
    try:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z0-9\s\?!.,\'"]', '', text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words or word in ['not', 'no', "n't"] and word != '']
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    except Exception as e:
        logger.error(f"Error in text preprocessing: {str(e)}")
        st.error(f"Error processing text: {str(e)}")
        return None

def main():
    st.title('Question Type Classification Using Machine Learning')
    
    # Initialize resources if not already done
    if not st.session_state.resources_loaded:
        with st.spinner('Loading resources...'):
            if setup_nltk():
                model_files = load_model_files()
                if model_files is not None:
                    st.session_state.model, st.session_state.tokenizer, \
                    st.session_state.label_encoder, st.session_state.maxlen = model_files
                    st.session_state.resources_loaded = True
                    st.success('Resources loaded successfully!')
                else:
                    st.error('Failed to load model files!')
                    return
    
    # Input field
    text = st.text_input("Enter your question:", key="input1")
    
    if text.strip():
        # Process and predict
        try:
            text_prepared = preprocess_text(text)
            if text_prepared is None:
                return
                
            sequence_testing = st.session_state.tokenizer.texts_to_sequences([text_prepared])
            padded_testing = pad_sequences(sequence_testing, 
                                         maxlen=st.session_state.maxlen, 
                                         padding='post')
            
            prediksi = st.session_state.model.predict(padded_testing, verbose=0)
            predicted_class = np.argmax(prediksi, axis=1)[0]
            predicted_label = st.session_state.label_encoder.inverse_transform([predicted_class])[0]
            
            # Display results
            st.write("Predicted Question Type:", predicted_label)
            
            # Display confidence scores
            st.write("Confidence Scores:")
            for idx, prob in enumerate(prediksi[0]):
                label = st.session_state.label_encoder.inverse_transform([idx])[0]
                st.write(f"{label}: {prob:.2%}")
                
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
