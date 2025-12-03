#!/usr/bin/env python3
"""
Text Classification System
Multi-class text classification for various categories
"""

import pandas as pd
import numpy as np
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
import pickle
import os

class TextClassifier:
    """
    Multi-class text classifier for various text categories
    Categories: Technology, Sports, Politics, Entertainment, Business, Health
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100)
        }
        self.best_model = None
        self.best_model_name = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.categories = ['Technology', 'Sports', 'Politics', 'Entertainment', 'Business', 'Health']
        self.trained = False
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
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
    
    def generate_sample_data(self):
        """Generate sample training data for different categories"""
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
        
        # Create DataFrame
        texts = []
        labels = []
        
        for category, sentences in data.items():
            for sentence in sentences:
                texts.append(sentence)
                labels.append(category)
        
        return pd.DataFrame({'text': texts, 'category': labels})
    
    def train_models(self, df=None):
        """Train multiple models and select the best one"""
        if df is None:
            df = self.generate_sample_data()
        
        print("Training text classification models...")
        print(f"Dataset size: {len(df)} samples")
        print(f"Categories: {df['category'].unique()}")
        
        # Preprocess texts
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
        )
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train and evaluate models
        best_accuracy = 0
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Train model
            model.fit(X_train_vec, y_train)
            
            # Predict
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[model_name] = {
                'accuracy': accuracy,
                'model': model,
                'classification_report': classification_report(y_test, y_pred),
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"{model_name} Accuracy: {accuracy:.4f}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = model_name
        
        print(f"\nBest model: {self.best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        # Print detailed results for best model
        print(f"\nDetailed Results for {self.best_model_name}:")
        print(results[self.best_model_name]['classification_report'])
        
        self.trained = True
        self.save_model()
        
        return results
    
    def predict(self, text):
        """Predict category for a single text"""
        if not self.trained:
            print("Model not trained. Training now...")
            self.train_models()
        
        processed_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([processed_text])
        
        prediction = self.best_model.predict(text_vec)[0]
        probabilities = self.best_model.predict_proba(text_vec)[0]
        
        # Get probability for each category
        category_probs = {}
        for i, category in enumerate(self.best_model.classes_):
            category_probs[category] = probabilities[i]
        
        return {
            'text': text,
            'predicted_category': prediction,
            'confidence': max(probabilities),
            'all_probabilities': category_probs,
            'model_used': self.best_model_name
        }
    
    def batch_predict(self, texts):
        """Predict categories for multiple texts"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def save_model(self):
        """Save trained model and vectorizer"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save best model
        with open('models/text_classifier.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save vectorizer
        with open('models/text_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'categories': list(self.best_model.classes_),
            'trained': self.trained
        }
        
        with open('models/classifier_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_model(self):
        """Load saved model and vectorizer"""
        try:
            # Load model
            with open('models/text_classifier.pkl', 'rb') as f:
                self.best_model = pickle.load(f)
            
            # Load vectorizer
            with open('models/text_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load metadata
            with open('models/classifier_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
                self.best_model_name = metadata['model_name']
                self.categories = metadata['categories']
                self.trained = metadata['trained']
            
            print("Text classifier loaded successfully!")
            return True
        except FileNotFoundError:
            print("No saved model found. Training new model...")
            return False

def main():
    """Test the text classifier"""
    classifier = TextClassifier()
    
    # Try to load existing model
    if not classifier.load_model():
        classifier.train_models()
    
    # Test sentences
    test_texts = [
        "The new iPhone features advanced AI capabilities and improved camera technology",
        "The basketball team won the championship with an outstanding performance",
        "The government announced new policies to address climate change",
        "The movie broke box office records and received critical acclaim",
        "The company's quarterly earnings exceeded analyst expectations",
        "New medical research shows promising results for treating diabetes"
    ]
    
    print("\n" + "="*60)
    print("TEXT CLASSIFICATION RESULTS")
    print("="*60)
    
    for text in test_texts:
        result = classifier.predict(text)
        print(f"\nText: {text}")
        print(f"Category: {result['predicted_category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Top 3 predictions:")
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, prob in sorted_probs:
            print(f"  {cat}: {prob:.3f}")

if __name__ == "__main__":
    main()