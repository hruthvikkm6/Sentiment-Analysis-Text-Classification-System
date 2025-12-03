#!/usr/bin/env python3
"""
Setup script to download required NLTK data
"""

import nltk
import ssl

def download_nltk_data():
    """Download required NLTK data packages"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    print("Downloading NLTK data...")
    
    # Download required datasets
    datasets = [
        'punkt',
        'vader_lexicon',
        'stopwords',
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4',
        'movie_reviews',
        'subjectivity'
    ]
    
    for dataset in datasets:
        try:
            nltk.download(dataset)
            print(f"✓ Downloaded {dataset}")
        except Exception as e:
            print(f"✗ Failed to download {dataset}: {e}")
    
    print("NLTK data download complete!")

if __name__ == "__main__":
    download_nltk_data()