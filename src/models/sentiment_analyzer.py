#!/usr/bin/env python3
"""
Advanced Sentiment Analysis Model
Combines multiple approaches for robust sentiment analysis
"""

import nltk
import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords, movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import os

class SentimentAnalyzer:
    """
    Advanced sentiment analyzer using multiple techniques:
    1. VADER Sentiment Analyzer
    2. TextBlob Polarity
    3. Custom Naive Bayes Model
    4. Rule-based Approach
    """
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.nb_model = MultinomialNB()
        self.model_trained = False
        
        # Emotion keywords for enhanced analysis
        self.positive_words = {
            'love', 'amazing', 'awesome', 'excellent', 'fantastic', 'great', 'wonderful',
            'brilliant', 'outstanding', 'perfect', 'superb', 'incredible', 'magnificent',
            'marvelous', 'spectacular', 'phenomenal', 'extraordinary', 'remarkable'
        }
        
        self.negative_words = {
            'hate', 'awful', 'terrible', 'horrible', 'disgusting', 'pathetic', 'worst',
            'dreadful', 'appalling', 'abysmal', 'atrocious', 'deplorable', 'despicable',
            'detestable', 'loathsome', 'repulsive', 'revolting', 'sickening'
        }
    
    def preprocess_text(self, text):
        """Clean and preprocess text for analysis"""
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
    
    def vader_analysis(self, text):
        """Analyze sentiment using VADER"""
        scores = self.vader_analyzer.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'neutral': scores['neu'],
            'negative': scores['neg'],
            'sentiment': 'positive' if scores['compound'] >= 0.05 else 'negative' if scores['compound'] <= -0.05 else 'neutral'
        }
    
    def textblob_analysis(self, text):
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': 'positive' if polarity > 0.1 else 'negative' if polarity < -0.1 else 'neutral'
        }
    
    def rule_based_analysis(self, text):
        """Rule-based sentiment analysis using keyword matching"""
        words = set(text.lower().split())
        
        positive_score = len(words.intersection(self.positive_words))
        negative_score = len(words.intersection(self.negative_words))
        
        if positive_score > negative_score:
            sentiment = 'positive'
            confidence = (positive_score / (positive_score + negative_score + 1))
        elif negative_score > positive_score:
            sentiment = 'negative'
            confidence = (negative_score / (positive_score + negative_score + 1))
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'sentiment': sentiment,
            'confidence': confidence
        }
    
    def train_naive_bayes(self):
        """Train Naive Bayes model on movie reviews dataset"""
        print("Training Naive Bayes model...")
        
        # Load movie reviews dataset
        documents = [
            (list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)
        ]
        
        # Prepare training data
        texts = []
        labels = []
        
        for words, label in documents:
            text = ' '.join(words)
            processed_text = self.preprocess_text(text)
            texts.append(processed_text)
            labels.append(label)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.nb_model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.nb_model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Naive Bayes Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.model_trained = True
        
        # Save model
        self.save_model()
        
        return accuracy
    
    def naive_bayes_predict(self, text):
        """Predict sentiment using trained Naive Bayes model"""
        if not self.model_trained:
            print("Model not trained. Training now...")
            self.train_naive_bayes()
        
        processed_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([processed_text])
        prediction = self.nb_model.predict(text_vec)[0]
        probability = max(self.nb_model.predict_proba(text_vec)[0])
        
        return {
            'sentiment': prediction,
            'confidence': probability
        }
    
    def comprehensive_analysis(self, text):
        """Perform comprehensive sentiment analysis using all methods"""
        if not text or not isinstance(text, str):
            return {'error': 'Invalid input text'}
        
        # Get results from all methods
        vader_result = self.vader_analysis(text)
        textblob_result = self.textblob_analysis(text)
        rule_result = self.rule_based_analysis(text)
        nb_result = self.naive_bayes_predict(text) if self.model_trained else {'sentiment': 'neutral', 'confidence': 0.5}
        
        # Ensemble voting
        sentiments = [
            vader_result['sentiment'],
            textblob_result['sentiment'],
            rule_result['sentiment'],
            nb_result['sentiment']
        ]
        
        # Count votes
        sentiment_counts = {sentiment: sentiments.count(sentiment) for sentiment in ['positive', 'negative', 'neutral']}
        final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        # Calculate confidence
        confidence = sentiment_counts[final_sentiment] / len(sentiments)
        
        return {
            'text': text,
            'final_sentiment': final_sentiment,
            'confidence': confidence,
            'details': {
                'vader': vader_result,
                'textblob': textblob_result,
                'rule_based': rule_result,
                'naive_bayes': nb_result
            },
            'voting_results': sentiment_counts
        }
    
    def batch_analysis(self, texts):
        """Analyze multiple texts"""
        results = []
        for text in texts:
            result = self.comprehensive_analysis(text)
            results.append(result)
        return results
    
    def save_model(self):
        """Save trained model and vectorizer"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        with open('models/nb_model.pkl', 'wb') as f:
            pickle.dump(self.nb_model, f)
        
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load_model(self):
        """Load saved model and vectorizer"""
        try:
            with open('models/nb_model.pkl', 'rb') as f:
                self.nb_model = pickle.load(f)
            
            with open('models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.model_trained = True
            print("Model loaded successfully!")
            return True
        except FileNotFoundError:
            print("No saved model found. Training new model...")
            return False

def main():
    """Test the sentiment analyzer"""
    analyzer = SentimentAnalyzer()
    
    # Try to load existing model
    if not analyzer.load_model():
        analyzer.train_naive_bayes()
    
    # Test sentences
    test_sentences = [
        "I love this movie! It's absolutely fantastic!",
        "This is the worst film I've ever seen. Terrible acting.",
        "The movie was okay, nothing special.",
        "Amazing cinematography and brilliant performances!",
        "I hate this boring and predictable plot."
    ]
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*60)
    
    for sentence in test_sentences:
        result = analyzer.comprehensive_analysis(sentence)
        print(f"\nText: {sentence}")
        print(f"Final Sentiment: {result['final_sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Voting: {result['voting_results']}")

if __name__ == "__main__":
    main()