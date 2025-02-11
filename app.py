# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from pathlib import Path

# Initialize NLTK
nltk_data_dir = Path("./nltk_data")
nltk_data_dir.mkdir(exist_ok=True)
nltk.data.path.append(str(nltk_data_dir))

# Download required NLTK data
for resource in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=str(nltk_data_dir))

# Safe load model files
def load_pickle(filename):
    try:
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None

# Load model and resources
@st.cache_resource
def load_resources():
    model = keras.models.load_model('sentimen_model.h5')
    tokenizer = load_pickle('tokenizer.pkl')
    label_encoder = load_pickle('label_encoderA.pkl')  # Note the correct filename
    maxlen = load_pickle('maxlen.pkl')
    return model, tokenizer, label_encoder, maxlen

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s\?!.,\'"]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words or word in ['not', 'no', "n't"]]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def main():
    st.title('Klasifikasi Jenis Pertanyaan Menggunakan Machine Learning')
    
    # Load resources
    model, tokenizer, label_encoder, maxlen = load_resources()
    
    if None in (model, tokenizer, label_encoder, maxlen):
        st.error("Failed to load required resources.")
        st.info("Make sure all required files exist: 'sentimen_model.h5', 'tokenizer.pkl', 'label_encoderA.pkl', 'maxlen.pkl'")
        return
    
    # Input
    text = st.text_input("Masukkan Pertanyaan:", key="input1")
    
    if text.strip():
        try:
            # Process input
            text_prepared = preprocess_text(text)
            sequence = tokenizer.texts_to_sequences([text_prepared])
            padded = pad_sequences(sequence, maxlen=maxlen, padding='post')
            
            # Predict
            prediction = model.predict(padded)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            
            # Show results
            st.write("Hasil Prediksi (Class):", predicted_label)
            confidence = float(prediction[0][predicted_class])
            st.write(f"Confidence Score: {confidence:.2%}")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
