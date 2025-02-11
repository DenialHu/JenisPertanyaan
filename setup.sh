#!/bin/bash

# Buat direktori nltk_data jika belum ada
mkdir -p nltk_data

# Unduh data NLTK ke direktori nltk_data
python3 -m nltk.downloader -d ./nltk_data punkt
