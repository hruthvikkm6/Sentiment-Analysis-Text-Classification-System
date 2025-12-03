#!/usr/bin/env python3
"""
Test suite for Sentiment Analysis & Text Classification System
"""

import sys
import os
import unittest
from datetime import datetime

# Add src to path
sys.path.append('src')

from models.sentiment_analyzer import SentimentAnalyzer
from models.text_classifier import TextClassifier
from utils.data_processor import DataProcessor, DataValidator

class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for sentiment analyzer"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.analyzer = SentimentAnalyzer()
        # Use sample data for testing to avoid long training
        print("Setting up sentiment analyzer for testing...")
    
    def test_preprocess_text(self):
        """Test text preprocessing"""
        text = "I LOVE this product!!! 😍 http://example.com @user #hashtag"
        processed = self.analyzer.preprocess_text(text)
        
        self.assertIsInstance(processed, str)
        self.assertNotIn("http://", processed)
        self.assertNotIn("@user", processed)
        self.assertNotIn("#hashtag", processed)
    
    def test_vader_analysis(self):
        """Test VADER sentiment analysis"""
        positive_text = "I absolutely love this amazing product!"
        negative_text = "I hate this terrible awful thing!"
        neutral_text = "This is a chair."
        
        pos_result = self.analyzer.vader_analysis(positive_text)
        neg_result = self.analyzer.vader_analysis(negative_text)
        neu_result = self.analyzer.vader_analysis(neutral_text)
        
        self.assertEqual(pos_result['sentiment'], 'positive')
        self.assertEqual(neg_result['sentiment'], 'negative')
        self.assertIn(neu_result['sentiment'], ['neutral', 'positive', 'negative'])
    
    def test_textblob_analysis(self):
        """Test TextBlob sentiment analysis"""
        text = "I love this product!"
        result = self.analyzer.textblob_analysis(text)
        
        self.assertIn('polarity', result)
        self.assertIn('subjectivity', result)
        self.assertIn('sentiment', result)
        self.assertIn(result['sentiment'], ['positive', 'negative', 'neutral'])
    
    def test_comprehensive_analysis(self):
        """Test comprehensive sentiment analysis"""
        text = "I love this amazing product! Best purchase ever!"
        result = self.analyzer.comprehensive_analysis(text)
        
        # Check structure
        self.assertIn('text', result)
        self.assertIn('final_sentiment', result)
        self.assertIn('confidence', result)
        self.assertIn('details', result)
        self.assertIn('voting_results', result)
        
        # Check data types
        self.assertIsInstance(result['confidence'], float)
        self.assertIn(result['final_sentiment'], ['positive', 'negative', 'neutral'])
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)
    
    def test_batch_analysis(self):
        """Test batch processing"""
        texts = [
            "I love this!",
            "I hate this!",
            "This is okay."
        ]
        
        results = self.analyzer.batch_analysis(texts)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('final_sentiment', result)
            self.assertIn('confidence', result)

class TestTextClassifier(unittest.TestCase):
    """Test cases for text classifier"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.classifier = TextClassifier()
        print("Setting up text classifier for testing...")
        # Train with sample data
        cls.classifier.train_models()
    
    def test_preprocess_text(self):
        """Test text preprocessing"""
        text = "NEW iPhone 15 with AMAZING camera! Check it out: http://example.com"
        processed = self.classifier.preprocess_text(text)
        
        self.assertIsInstance(processed, str)
        self.assertNotIn("http://", processed)
        # Should be lowercase
        self.assertEqual(processed, processed.lower())
    
    def test_predict(self):
        """Test text classification prediction"""
        tech_text = "The new iPhone features advanced AI capabilities and machine learning algorithms."
        result = self.classifier.predict(tech_text)
        
        # Check structure
        self.assertIn('text', result)
        self.assertIn('predicted_category', result)
        self.assertIn('confidence', result)
        self.assertIn('all_probabilities', result)
        
        # Check data types
        self.assertIsInstance(result['confidence'], float)
        self.assertIn(result['predicted_category'], self.classifier.categories)
    
    def test_batch_predict(self):
        """Test batch classification"""
        texts = [
            "New smartphone with AI technology",
            "Basketball team wins championship",
            "Government announces new policy"
        ]
        
        results = self.classifier.batch_predict(texts)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('predicted_category', result)
            self.assertIn('confidence', result)

class TestDataProcessor(unittest.TestCase):
    """Test cases for data processor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor()
        
        # Sample results for testing
        self.sample_results = [
            {
                'text': 'I love this!',
                'final_sentiment': 'positive',
                'confidence': 0.9,
                'details': {
                    'vader': {'sentiment': 'positive', 'compound': 0.6},
                    'textblob': {'sentiment': 'positive', 'polarity': 0.5},
                    'rule_based': {'sentiment': 'positive'},
                    'naive_bayes': {'sentiment': 'positive'}
                }
            },
            {
                'text': 'I hate this!',
                'final_sentiment': 'negative',
                'confidence': 0.8,
                'details': {
                    'vader': {'sentiment': 'negative', 'compound': -0.7},
                    'textblob': {'sentiment': 'negative', 'polarity': -0.6},
                    'rule_based': {'sentiment': 'negative'},
                    'naive_bayes': {'sentiment': 'negative'}
                }
            }
        ]
    
    def test_generate_statistics(self):
        """Test statistics generation"""
        stats = self.processor.generate_statistics(self.sample_results)
        
        self.assertIn('total_analyzed', stats)
        self.assertIn('sentiment_distribution', stats)
        self.assertIn('confidence_stats', stats)
        self.assertIn('positive_percentage', stats)
        
        self.assertEqual(stats['total_analyzed'], 2)
    
    def test_save_results_to_json(self):
        """Test JSON export"""
        filename = 'tmp_test_results.json'
        success = self.processor.save_results_to_json(self.sample_results, filename)
        
        self.assertTrue(success)
        
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)
    
    def test_save_results_to_csv(self):
        """Test CSV export"""
        filename = 'tmp_test_results.csv'
        success = self.processor.save_results_to_csv(self.sample_results, filename)
        
        self.assertTrue(success)
        
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)

class TestDataValidator(unittest.TestCase):
    """Test cases for data validator"""
    
    def test_validate_text_input(self):
        """Test text input validation"""
        # Valid cases
        valid, msg = DataValidator.validate_text_input("Hello world!")
        self.assertTrue(valid)
        
        # Invalid cases
        invalid_cases = [
            "",  # Empty string
            None,  # None
            123,  # Not string
            "   ",  # Only whitespace
            "a" * 10001  # Too long
        ]
        
        for case in invalid_cases:
            valid, msg = DataValidator.validate_text_input(case)
            self.assertFalse(valid)
    
    def test_validate_batch_input(self):
        """Test batch input validation"""
        # Valid case
        valid_texts = ["Hello", "World", "Test"]
        valid, msg = DataValidator.validate_batch_input(valid_texts)
        self.assertTrue(valid)
        
        # Invalid cases
        invalid_cases = [
            [],  # Empty list
            None,  # None
            "not a list",  # Not a list
            [""] * 101,  # Too many items
        ]
        
        for case in invalid_cases:
            valid, msg = DataValidator.validate_batch_input(case)
            self.assertFalse(valid)

def run_performance_tests():
    """Run performance tests"""
    print("\n" + "="*50)
    print("PERFORMANCE TESTS")
    print("="*50)
    
    analyzer = SentimentAnalyzer()
    classifier = TextClassifier()
    
    # Test single analysis performance
    import time
    
    text = "I absolutely love this amazing product! The quality is outstanding and the customer service was exceptional."
    
    # Sentiment analysis timing
    start_time = time.time()
    result = analyzer.comprehensive_analysis(text)
    sentiment_time = time.time() - start_time
    
    print(f"Sentiment Analysis: {sentiment_time:.3f} seconds")
    
    # Classification timing (if trained)
    try:
        start_time = time.time()
        result = classifier.predict(text)
        classification_time = time.time() - start_time
        print(f"Text Classification: {classification_time:.3f} seconds")
    except:
        print("Text Classification: Model not trained")
    
    # Batch processing timing
    batch_texts = [
        "I love this product!",
        "This is terrible.",
        "The weather is nice.",
        "Amazing technology breakthrough!",
        "Basketball game was exciting."
    ] * 10  # 50 texts total
    
    start_time = time.time()
    results = analyzer.batch_analysis(batch_texts)
    batch_time = time.time() - start_time
    
    print(f"Batch Processing (50 texts): {batch_time:.3f} seconds ({50/batch_time:.1f} texts/second)")

def main():
    """Run all tests"""
    print("Sentiment Analysis & Text Classification System - Test Suite")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestTextClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance tests
    run_performance_tests()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return len(result.failures) + len(result.errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)