#!/usr/bin/env python3
"""
Sentiment Analysis Model Training Script
Train from scratch using custom dataset while maintaining compatibility with existing system
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords, movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('movie_reviews', quiet=True)

class SentimentModelTrainer:
    """
    Train sentiment analysis model from scratch using custom dataset
    Produces models compatible with existing SentimentAnalyzer class
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.nb_model = MultinomialNB()
        
    def preprocess_text(self, text):
        """Clean and preprocess text for training"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep emoticons
        text = re.sub(r'[^a-zA-Z0-9\s:);(]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def load_custom_dataset(self, file_path, text_column='text', label_column='sentiment'):
        """
        Load custom dataset from CSV file
        
        Expected format:
        - CSV file with columns for text and sentiment
        - Sentiment values should be 'positive' or 'negative'
        
        Args:
            file_path (str): Path to CSV file
            text_column (str): Name of text column (default: 'text')
            label_column (str): Name of label column (default: 'sentiment')
        
        Returns:
            pandas.DataFrame: Loaded and validated dataset
        """
        print(f"Loading dataset from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Validate required columns
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in dataset")
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in dataset")
            
            # Clean data
            df = df.dropna(subset=[text_column, label_column])
            
            # Validate sentiment labels
            valid_labels = {'positive', 'negative', 'pos', 'neg'}
            df[label_column] = df[label_column].astype(str).str.lower()
            
            # Standardize labels
            df[label_column] = df[label_column].replace({'pos': 'positive', 'neg': 'negative'})
            
            # Filter valid labels
            df = df[df[label_column].isin(['positive', 'negative'])]
            
            if len(df) == 0:
                raise ValueError("No valid sentiment labels found. Labels should be 'positive' or 'negative'")
            
            print(f"Dataset after cleaning: {len(df)} samples")
            print(f"Label distribution:\n{df[label_column].value_counts()}")
            
            return df[[text_column, label_column]].rename(columns={
                text_column: 'text',
                label_column: 'sentiment'
            })
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def load_default_dataset(self):
        """Load the default movie reviews dataset from NLTK"""
        print("Loading default movie reviews dataset from NLTK...")
        
        documents = [
            (list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)
        ]
        
        texts = []
        labels = []
        
        for words, label in documents:
            text = ' '.join(words)
            texts.append(text)
            labels.append(label)
        
        df = pd.DataFrame({
            'text': texts,
            'sentiment': labels
        })
        
        print(f"Default dataset loaded. Shape: {df.shape}")
        print(f"Label distribution:\n{df['sentiment'].value_counts()}")
        
        return df
    
    def train_model(self, dataset_path=None, text_column='text', label_column='sentiment', 
                   test_size=0.2, random_state=42, save_model=True):
        """
        Train sentiment analysis model
        
        Args:
            dataset_path (str): Path to custom CSV dataset (optional)
            text_column (str): Name of text column in CSV
            label_column (str): Name of sentiment label column in CSV
            test_size (float): Proportion of dataset to use for testing
            random_state (int): Random state for reproducibility
            save_model (bool): Whether to save the trained model
        
        Returns:
            dict: Training results including accuracy and classification report
        """
        print("="*60)
        print("TRAINING SENTIMENT ANALYSIS MODEL")
        print("="*60)
        
        # Load dataset
        if dataset_path:
            df = self.load_custom_dataset(dataset_path, text_column, label_column)
            if df is None:
                print("Failed to load custom dataset. Using default dataset instead.")
                df = self.load_default_dataset()
        else:
            df = self.load_default_dataset()
        
        # Preprocess texts
        print("\nPreprocessing texts...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty processed texts
        df = df[df['processed_text'].str.len() > 0]
        print(f"Dataset after preprocessing: {len(df)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['sentiment'], 
            test_size=test_size, random_state=random_state, stratify=df['sentiment']
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Vectorize text
        print("Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Feature vector shape: {X_train_vec.shape}")
        
        # Train model
        print("Training Naive Bayes model...")
        self.nb_model.fit(X_train_vec, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        y_pred = self.nb_model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Results
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_count': X_train_vec.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"\n{'='*60}")
        print("TRAINING RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Feature count: {results['feature_count']}")
        print(f"Training samples: {results['training_samples']}")
        print(f"Test samples: {results['test_samples']}")
        print(f"\nClassification Report:")
        print(results['classification_report'])
        print(f"\nConfusion Matrix:")
        print(results['confusion_matrix'])
        
        # Save model
        if save_model:
            self.save_model()
            print(f"\nModel saved successfully!")
        
        return results
    
    def save_model(self):
        """Save trained model and vectorizer to match existing system structure"""
        if not os.path.exists('models'):
            os.makedirs('models')
            print("Created models directory")
        
        # Save model (compatible with existing SentimentAnalyzer.load_model())
        model_path = 'models/nb_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.nb_model, f)
        print(f"Naive Bayes model saved to: {model_path}")
        
        # Save vectorizer (compatible with existing SentimentAnalyzer.load_model())
        vectorizer_path = 'models/vectorizer.pkl'
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Vectorizer saved to: {vectorizer_path}")
    
    def test_model_compatibility(self):
        """Test if trained model is compatible with existing SentimentAnalyzer"""
        try:
            # Try to load the model using the existing structure
            with open('models/nb_model.pkl', 'rb') as f:
                loaded_model = pickle.load(f)
            
            with open('models/vectorizer.pkl', 'rb') as f:
                loaded_vectorizer = pickle.load(f)
            
            # Test prediction
            test_text = "This is a great movie!"
            processed_text = self.preprocess_text(test_text)
            text_vec = loaded_vectorizer.transform([processed_text])
            prediction = loaded_model.predict(text_vec)[0]
            probability = max(loaded_model.predict_proba(text_vec)[0])
            
            print(f"\nCompatibility Test:")
            print(f"Test text: {test_text}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {probability:.4f}")
            print("✓ Model is compatible with existing SentimentAnalyzer!")
            
            return True
        except Exception as e:
            print(f"✗ Compatibility test failed: {e}")
            return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Train Sentiment Analysis Model')
    parser.add_argument('--dataset', type=str, help='Path to custom CSV dataset')
    parser.add_argument('--text_column', type=str, default='text', help='Name of text column')
    parser.add_argument('--label_column', type=str, default='sentiment', help='Name of label column')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--no_save', action='store_true', help='Do not save the model')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SentimentModelTrainer()
    
    # Train model
    results = trainer.train_model(
        dataset_path=args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        test_size=args.test_size,
        random_state=args.random_state,
        save_model=not args.no_save
    )
    
    # Test compatibility
    if not args.no_save:
        trainer.test_model_compatibility()

if __name__ == "__main__":
    main()