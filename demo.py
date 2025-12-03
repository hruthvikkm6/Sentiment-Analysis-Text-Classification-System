#!/usr/bin/env python3
"""
Demonstration script for Sentiment Analysis & Text Classification System
This script showcases all the features of the system
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

from models.sentiment_analyzer import SentimentAnalyzer
from models.text_classifier import TextClassifier
from utils.data_processor import DataProcessor

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_subheader(title):
    """Print a formatted subheader"""
    print(f"\n--- {title} ---")

def demo_sentiment_analysis():
    """Demonstrate sentiment analysis capabilities"""
    print_header("SENTIMENT ANALYSIS DEMONSTRATION")
    
    analyzer = SentimentAnalyzer()
    
    # Try to load existing model, otherwise train
    if not analyzer.load_model():
        print("Training sentiment analysis model (this may take a moment)...")
        analyzer.train_naive_bayes()
    
    # Demo texts
    demo_texts = [
        {
            'text': "I absolutely love this new smartphone! The camera quality is outstanding and the battery life is amazing!",
            'expected': 'positive'
        },
        {
            'text': "This movie was terrible. Worst acting I've ever seen and the plot made no sense.",
            'expected': 'negative'
        },
        {
            'text': "The weather today is cloudy with a chance of rain.",
            'expected': 'neutral'
        },
        {
            'text': "OMG this pizza is INCREDIBLE!!! 🍕😍 Best food ever!!!",
            'expected': 'positive'
        },
        {
            'text': "I'm so disappointed with this purchase. Complete waste of money.",
            'expected': 'negative'
        }
    ]
    
    results = []
    
    for i, item in enumerate(demo_texts, 1):
        text = item['text']
        expected = item['expected']
        
        print_subheader(f"Example {i}")
        print(f"Text: {text}")
        print(f"Expected: {expected}")
        
        result = analyzer.comprehensive_analysis(text)
        predicted = result['final_sentiment']
        confidence = result['confidence']
        
        print(f"Predicted: {predicted} (Confidence: {confidence:.3f})")
        
        # Show detailed breakdown
        print("Detailed Results:")
        print(f"  VADER: {result['details']['vader']['sentiment']} ({result['details']['vader']['compound']:.3f})")
        print(f"  TextBlob: {result['details']['textblob']['sentiment']} ({result['details']['textblob']['polarity']:.3f})")
        print(f"  Rule-based: {result['details']['rule_based']['sentiment']}")
        print(f"  Naive Bayes: {result['details']['naive_bayes']['sentiment']} ({result['details']['naive_bayes']['confidence']:.3f})")
        
        # Voting breakdown
        print("Voting Results:")
        for sentiment, votes in result['voting_results'].items():
            print(f"  {sentiment}: {votes}/4 votes")
        
        # Accuracy check
        is_correct = predicted == expected
        print(f"Accuracy: {'✓ Correct' if is_correct else '✗ Incorrect'}")
        
        results.append({
            'text': text,
            'expected': expected,
            'predicted': predicted,
            'confidence': confidence,
            'correct': is_correct
        })
        
        print()
    
    # Summary
    correct_predictions = sum(1 for r in results if r['correct'])
    accuracy = correct_predictions / len(results)
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    print_subheader("SENTIMENT ANALYSIS SUMMARY")
    print(f"Total Examples: {len(results)}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    return results

def demo_text_classification():
    """Demonstrate text classification capabilities"""
    print_header("TEXT CLASSIFICATION DEMONSTRATION")
    
    classifier = TextClassifier()
    
    # Try to load existing model, otherwise train
    if not classifier.load_model():
        print("Training text classification model...")
        classifier.train_models()
    
    # Demo texts for different categories
    demo_texts = [
        {
            'text': "Apple announces new iPhone with revolutionary AI-powered camera and enhanced machine learning capabilities.",
            'expected': 'Technology'
        },
        {
            'text': "The Lakers defeated the Warriors 112-108 in an exciting NBA playoff game last night.",
            'expected': 'Sports'
        },
        {
            'text': "The president signed new legislation addressing climate change and renewable energy initiatives.",
            'expected': 'Politics'
        },
        {
            'text': "The new Marvel movie broke box office records, earning $200 million in its opening weekend.",
            'expected': 'Entertainment'
        },
        {
            'text': "Tesla's stock price surged 15% after reporting better-than-expected quarterly earnings.",
            'expected': 'Business'
        },
        {
            'text': "Medical researchers published breakthrough findings on a new treatment for diabetes patients.",
            'expected': 'Health'
        }
    ]
    
    results = []
    
    for i, item in enumerate(demo_texts, 1):
        text = item['text']
        expected = item['expected']
        
        print_subheader(f"Example {i}")
        print(f"Text: {text}")
        print(f"Expected: {expected}")
        
        result = classifier.predict(text)
        predicted = result['predicted_category']
        confidence = result['confidence']
        
        print(f"Predicted: {predicted} (Confidence: {confidence:.3f})")
        print(f"Model Used: {result['model_used']}")
        
        # Show top 3 predictions
        print("Top 3 Predictions:")
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        for category, prob in sorted_probs:
            print(f"  {category}: {prob:.3f}")
        
        # Accuracy check
        is_correct = predicted == expected
        print(f"Accuracy: {'✓ Correct' if is_correct else '✗ Incorrect'}")
        
        results.append({
            'text': text,
            'expected': expected,
            'predicted': predicted,
            'confidence': confidence,
            'correct': is_correct
        })
        
        print()
    
    # Summary
    correct_predictions = sum(1 for r in results if r['correct'])
    accuracy = correct_predictions / len(results)
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    print_subheader("TEXT CLASSIFICATION SUMMARY")
    print(f"Total Examples: {len(results)}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    # Category distribution
    category_counts = {}
    for result in results:
        cat = result['predicted']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nPredicted Category Distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
    
    return results

def demo_data_processing():
    """Demonstrate data processing and visualization capabilities"""
    print_header("DATA PROCESSING DEMONSTRATION")
    
    processor = DataProcessor()
    
    # Create sample results for demonstration
    sample_results = [
        {
            'text': 'I love this amazing product!',
            'final_sentiment': 'positive',
            'confidence': 0.92,
            'details': {
                'vader': {'sentiment': 'positive', 'compound': 0.7, 'positive': 0.8, 'neutral': 0.2, 'negative': 0.0},
                'textblob': {'sentiment': 'positive', 'polarity': 0.6, 'subjectivity': 0.8},
                'rule_based': {'sentiment': 'positive', 'positive_score': 2, 'negative_score': 0},
                'naive_bayes': {'sentiment': 'positive', 'confidence': 0.85}
            },
            'voting_results': {'positive': 4, 'negative': 0, 'neutral': 0}
        },
        {
            'text': 'This product is terrible and disappointing.',
            'final_sentiment': 'negative',
            'confidence': 0.88,
            'details': {
                'vader': {'sentiment': 'negative', 'compound': -0.8, 'positive': 0.0, 'neutral': 0.1, 'negative': 0.9},
                'textblob': {'sentiment': 'negative', 'polarity': -0.7, 'subjectivity': 0.9},
                'rule_based': {'sentiment': 'negative', 'positive_score': 0, 'negative_score': 2},
                'naive_bayes': {'sentiment': 'negative', 'confidence': 0.82}
            },
            'voting_results': {'positive': 0, 'negative': 4, 'neutral': 0}
        },
        {
            'text': 'The weather is okay today.',
            'final_sentiment': 'neutral',
            'confidence': 0.65,
            'details': {
                'vader': {'sentiment': 'neutral', 'compound': 0.1, 'positive': 0.3, 'neutral': 0.7, 'negative': 0.0},
                'textblob': {'sentiment': 'neutral', 'polarity': 0.0, 'subjectivity': 0.3},
                'rule_based': {'sentiment': 'neutral', 'positive_score': 0, 'negative_score': 0},
                'naive_bayes': {'sentiment': 'neutral', 'confidence': 0.6}
            },
            'voting_results': {'positive': 0, 'negative': 0, 'neutral': 4}
        }
    ]
    
    print_subheader("Statistical Analysis")
    
    # Generate statistics
    stats = processor.generate_statistics(sample_results)
    
    print("Dataset Statistics:")
    print(f"  Total Analyzed: {stats['total_analyzed']}")
    print(f"  Positive: {stats['positive_percentage']:.1f}%")
    print(f"  Negative: {stats['negative_percentage']:.1f}%")
    print(f"  Neutral: {stats['neutral_percentage']:.1f}%")
    print(f"  Average Confidence: {stats['confidence_stats']['mean']:.3f}")
    print(f"  Confidence Range: {stats['confidence_stats']['min']:.3f} - {stats['confidence_stats']['max']:.3f}")
    
    print_subheader("Data Export")
    
    # Test data export
    csv_success = processor.save_results_to_csv(sample_results, 'tmp_demo_results.csv')
    json_success = processor.save_results_to_json(sample_results, 'tmp_demo_results.json')
    
    print(f"CSV Export: {'✓ Success' if csv_success else '✗ Failed'}")
    print(f"JSON Export: {'✓ Success' if json_success else '✗ Failed'}")
    
    # Clean up temporary files
    for filename in ['tmp_demo_results.csv', 'tmp_demo_results.json']:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Cleaned up: {filename}")

def demo_batch_processing():
    """Demonstrate batch processing capabilities"""
    print_header("BATCH PROCESSING DEMONSTRATION")
    
    analyzer = SentimentAnalyzer()
    
    # Batch of texts for processing
    batch_texts = [
        "I absolutely love this new phone!",
        "The service was terrible and slow.",
        "It's an okay product, nothing special.",
        "Amazing quality and fast delivery!",
        "Worst purchase I've ever made.",
        "The weather is nice today.",
        "Fantastic customer support experience!",
        "Very disappointed with this item."
    ]
    
    print(f"Processing {len(batch_texts)} texts...")
    print()
    
    # Process batch
    results = analyzer.batch_analysis(batch_texts)
    
    # Display results
    for i, result in enumerate(results, 1):
        text = result['text']
        sentiment = result['final_sentiment']
        confidence = result['confidence']
        
        print(f"{i:2d}. {text}")
        print(f"    → {sentiment.upper()} (confidence: {confidence:.3f})")
        print()
    
    # Batch statistics
    sentiments = [r['final_sentiment'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    sentiment_counts = {s: sentiments.count(s) for s in ['positive', 'negative', 'neutral']}
    avg_confidence = sum(confidences) / len(confidences)
    
    print_subheader("BATCH PROCESSING SUMMARY")
    print(f"Total Processed: {len(results)}")
    print("Sentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    print(f"Average Confidence: {avg_confidence:.3f}")

def main():
    """Run all demonstrations"""
    print("🧠 SENTIMENT ANALYSIS & TEXT CLASSIFICATION SYSTEM")
    print("📊 COMPREHENSIVE DEMONSTRATION")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run demonstrations
        sentiment_results = demo_sentiment_analysis()
        classification_results = demo_text_classification()
        demo_data_processing()
        demo_batch_processing()
        
        # Overall summary
        print_header("DEMONSTRATION SUMMARY")
        
        print("✅ All demonstrations completed successfully!")
        print()
        print("System Capabilities Demonstrated:")
        print("  ✓ Multi-algorithm sentiment analysis")
        print("  ✓ Text classification into 6 categories")
        print("  ✓ Batch processing of multiple texts")
        print("  ✓ Statistical analysis and reporting")
        print("  ✓ Data export in multiple formats")
        print("  ✓ Comprehensive confidence scoring")
        print()
        print("Next Steps:")
        print("  1. Start the web application: python app.py")
        print("  2. Try the command-line interface: python sentiment_analyzer.py --help")
        print("  3. Run the test suite: python run_tests.py")
        print("  4. Explore the web interface at http://localhost:5000")
        
        # Performance summary
        total_examples = len(sentiment_results) + len(classification_results)
        sentiment_accuracy = sum(1 for r in sentiment_results if r['correct']) / len(sentiment_results)
        classification_accuracy = sum(1 for r in classification_results if r['correct']) / len(classification_results)
        
        print()
        print("Performance Summary:")
        print(f"  Total Examples Analyzed: {total_examples}")
        print(f"  Sentiment Analysis Accuracy: {sentiment_accuracy:.1%}")
        print(f"  Text Classification Accuracy: {classification_accuracy:.1%}")
        print(f"  Overall System Accuracy: {(sentiment_accuracy + classification_accuracy) / 2:.1%}")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎉 Demonstration completed successfully!' if success else '❌ Demonstration failed.'}")
    sys.exit(0 if success else 1)