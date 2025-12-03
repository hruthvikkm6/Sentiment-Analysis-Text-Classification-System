#!/usr/bin/env python3
"""
Text Classification Model Training Script
Train from scratch using custom dataset while maintaining compatibility with existing system
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextClassifierTrainer:
    """
    Train text classification model from scratch using custom dataset
    Produces models compatible with existing TextClassifier class
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.categories = []
        
    def preprocess_text(self, text):
        """Clean and preprocess text for training"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails, special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def load_custom_dataset(self, file_path, text_column='text', label_column='category'):
        """
        Load custom dataset from CSV file
        
        Expected format:
        - CSV file with columns for text and category
        - Category values can be any string labels
        
        Args:
            file_path (str): Path to CSV file
            text_column (str): Name of text column (default: 'text')
            label_column (str): Name of category column (default: 'category')
        
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
            
            # Clean and standardize labels
            df[label_column] = df[label_column].astype(str).str.strip()
            
            if len(df) == 0:
                raise ValueError("No valid data found after cleaning")
            
            # Get unique categories
            categories = df[label_column].unique()
            print(f"Found categories: {categories}")
            
            # Check minimum samples per category
            category_counts = df[label_column].value_counts()
            min_samples = category_counts.min()
            if min_samples < 2:
                print("Warning: Some categories have very few samples. Consider adding more data.")
            
            print(f"Dataset after cleaning: {len(df)} samples")
            print(f"Category distribution:\n{category_counts}")
            
            return df[[text_column, label_column]].rename(columns={
                text_column: 'text',
                label_column: 'category'
            })
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def generate_default_dataset(self):
        """Generate default sample dataset for demonstration"""
        print("Generating default sample dataset...")
        
        data = {
            'Technology': [
                "Artificial intelligence and machine learning are transforming the tech industry",
                "New smartphone with advanced camera technology released",
                "Cloud computing services gaining popularity among businesses",
                "Cybersecurity threats increasing with digital transformation",
                "Software development trends and programming languages",
                "Internet of Things devices connecting smart homes",
                "Big data analytics driving business decisions",
                "Blockchain technology revolutionizing finance",
                "Virtual reality gaming experiences improving",
                "Quantum computing breakthrough announced by tech giants"
            ],
            'Sports': [
                "Football championship final match results announced",
                "Basketball player breaks scoring record in playoff game",
                "Tennis tournament semifinals feature exciting matches",
                "Soccer world cup preparation underway for national teams",
                "Olympic swimming records broken at international competition",
                "Baseball season statistics show impressive team performance",
                "Golf tournament winner celebrates major victory",
                "Cricket match highlights showcase outstanding bowling",
                "Hockey playoffs intensify with overtime victories",
                "Marathon runners prepare for upcoming city race"
            ],
            'Politics': [
                "Election campaign promises focus on economic recovery",
                "Government policy changes affect healthcare system",
                "International diplomatic relations strengthen through summit",
                "Legislative session addresses climate change initiatives",
                "Political debate highlights education reform proposals",
                "Voting rights legislation discussed in parliamentary session",
                "Foreign policy decisions impact trade agreements",
                "Local government budget allocation prioritizes infrastructure",
                "Constitutional amendment process begins for civil rights",
                "Political party leadership changes announced"
            ],
            'Entertainment': [
                "New movie release breaks box office records",
                "Television series finale attracts millions of viewers",
                "Music album tops charts worldwide",
                "Celebrity couple announces engagement at awards ceremony",
                "Film festival showcases independent cinema",
                "Concert tour dates announced for popular band",
                "Streaming platform launches original content series",
                "Theater production receives standing ovation",
                "Video game sequel surpasses sales expectations",
                "Documentary film wins prestigious award"
            ],
            'Business': [
                "Company merger creates industry leader",
                "Startup receives significant venture capital funding",
                "Stock market reaches new highs amid economic growth",
                "Corporate earnings report exceeds analyst expectations",
                "Business expansion plans include international markets",
                "Entrepreneurship program supports small business development",
                "Economic forecast predicts continued market stability",
                "Industry conference showcases innovation trends",
                "Supply chain disruptions affect global commerce",
                "Investment strategy focuses on sustainable companies"
            ],
            'Health': [
                "Medical breakthrough offers hope for cancer treatment",
                "Public health campaign promotes vaccination awareness",
                "Nutrition research reveals benefits of balanced diet",
                "Mental health support services expand in communities",
                "Exercise program shows positive results for seniors",
                "Healthcare technology improves patient outcomes",
                "Pandemic response measures updated based on data",
                "Medical device innovation enhances surgical procedures",
                "Health insurance coverage expands for preventive care",
                "Wellness program implementation increases employee satisfaction"
            ]
        }
        
        texts = []
        labels = []
        
        for category, sentences in data.items():
            for sentence in sentences:
                texts.append(sentence)
                labels.append(category)
        
        df = pd.DataFrame({'text': texts, 'category': labels})
        
        print(f"Default dataset generated. Shape: {df.shape}")
        print(f"Categories: {df['category'].unique()}")
        print(f"Category distribution:\n{df['category'].value_counts()}")
        
        return df
    
    def train_models(self, dataset_path=None, text_column='text', label_column='category',
                    test_size=0.2, random_state=42, save_model=True):
        """
        Train text classification models and select the best one
        
        Args:
            dataset_path (str): Path to custom CSV dataset (optional)
            text_column (str): Name of text column in CSV
            label_column (str): Name of category label column in CSV
            test_size (float): Proportion of dataset to use for testing
            random_state (int): Random state for reproducibility
            save_model (bool): Whether to save the trained model
        
        Returns:
            dict: Training results for all models
        """
        print("="*60)
        print("TRAINING TEXT CLASSIFICATION MODELS")
        print("="*60)
        
        # Load dataset
        if dataset_path:
            df = self.load_custom_dataset(dataset_path, text_column, label_column)
            if df is None:
                print("Failed to load custom dataset. Using default dataset instead.")
                df = self.generate_default_dataset()
        else:
            df = self.generate_default_dataset()
        
        # Store categories
        self.categories = sorted(df['category'].unique())
        print(f"\nCategories to classify: {self.categories}")
        
        # Preprocess texts
        print("Preprocessing texts...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty processed texts
        df = df[df['processed_text'].str.len() > 0]
        print(f"Dataset after preprocessing: {len(df)} samples")
        
        # Check if we have enough samples for train/test split
        min_category_count = df['category'].value_counts().min()
        if min_category_count < 2:
            print("Error: Some categories have too few samples for train/test split")
            return None
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'], df['category'], 
                test_size=test_size, random_state=random_state, 
                stratify=df['category']
            )
        except ValueError as e:
            print(f"Stratification failed: {e}")
            # Try without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'], df['category'], 
                test_size=test_size, random_state=random_state
            )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Vectorize text
        print("Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Feature vector shape: {X_train_vec.shape}")
        
        # Train and evaluate models
        best_accuracy = 0
        results = {}
        
        print(f"\n{'='*40}")
        print("TRAINING INDIVIDUAL MODELS")
        print(f"{'='*40}")
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Train model
                model.fit(X_train_vec, y_train)
                
                # Predict
                y_pred = model.predict(X_test_vec)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'model': model,
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                print(f"{model_name} Accuracy: {accuracy:.4f}")
                
                # Track best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.best_model = model
                    self.best_model_name = model_name
                    
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        if self.best_model is None:
            print("Error: No models were trained successfully")
            return results
        
        print(f"\n{'='*60}")
        print("TRAINING RESULTS")
        print(f"{'='*60}")
        print(f"Best model: {self.best_model_name} (Accuracy: {best_accuracy:.4f})")
        print(f"Feature count: {X_train_vec.shape[1]}")
        print(f"Categories: {len(self.categories)}")
        
        # Print detailed results for best model
        print(f"\nDetailed Results for {self.best_model_name}:")
        print(results[self.best_model_name]['classification_report'])
        
        print(f"\nConfusion Matrix:")
        print(results[self.best_model_name]['confusion_matrix'])
        
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
        
        # Save best model (compatible with existing TextClassifier.load_model())
        model_path = 'models/text_classifier.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Text classifier saved to: {model_path}")
        
        # Save vectorizer (compatible with existing TextClassifier.load_model())
        vectorizer_path = 'models/text_vectorizer.pkl'
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Vectorizer saved to: {vectorizer_path}")
        
        # Save model metadata (compatible with existing TextClassifier.load_model())
        metadata = {
            'model_name': self.best_model_name,
            'categories': list(self.best_model.classes_),
            'trained': True
        }
        
        metadata_path = 'models/classifier_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to: {metadata_path}")
    
    def test_model_compatibility(self):
        """Test if trained model is compatible with existing TextClassifier"""
        try:
            # Try to load the model using the existing structure
            with open('models/text_classifier.pkl', 'rb') as f:
                loaded_model = pickle.load(f)
            
            with open('models/text_vectorizer.pkl', 'rb') as f:
                loaded_vectorizer = pickle.load(f)
            
            with open('models/classifier_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            # Test prediction
            test_text = "The new iPhone features advanced AI capabilities"
            processed_text = self.preprocess_text(test_text)
            text_vec = loaded_vectorizer.transform([processed_text])
            prediction = loaded_model.predict(text_vec)[0]
            probabilities = loaded_model.predict_proba(text_vec)[0]
            
            print(f"\nCompatibility Test:")
            print(f"Test text: {test_text}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {max(probabilities):.4f}")
            print(f"Model: {metadata['model_name']}")
            print(f"Categories: {metadata['categories']}")
            print("✓ Model is compatible with existing TextClassifier!")
            
            return True
        except Exception as e:
            print(f"✗ Compatibility test failed: {e}")
            return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Train Text Classification Model')
    parser.add_argument('--dataset', type=str, help='Path to custom CSV dataset')
    parser.add_argument('--text_column', type=str, default='text', help='Name of text column')
    parser.add_argument('--label_column', type=str, default='category', help='Name of category column')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--no_save', action='store_true', help='Do not save the model')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = TextClassifierTrainer()
    
    # Train models
    results = trainer.train_models(
        dataset_path=args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        test_size=args.test_size,
        random_state=args.random_state,
        save_model=not args.no_save
    )
    
    # Test compatibility
    if not args.no_save and results:
        trainer.test_model_compatibility()

if __name__ == "__main__":
    main()