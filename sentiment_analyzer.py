#!/usr/bin/env python3
"""
Command-line interface for Sentiment Analysis & Text Classification System
Usage: python sentiment_analyzer.py "Your text here"
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

from models.sentiment_analyzer import SentimentAnalyzer
from models.text_classifier import TextClassifier
from utils.data_processor import DataProcessor, DataValidator

def main():
    parser = argparse.ArgumentParser(
        description='Sentiment Analysis & Text Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python sentiment_analyzer.py "I love this product!"
  python sentiment_analyzer.py --classify "New AI breakthrough in technology"
  python sentiment_analyzer.py --batch texts.txt
  python sentiment_analyzer.py --interactive
        '''
    )
    
    parser.add_argument('text', nargs='?', help='Text to analyze')
    parser.add_argument('--classify', '-c', action='store_true', 
                       help='Perform text classification instead of sentiment analysis')
    parser.add_argument('--batch', '-b', metavar='FILE', 
                       help='Process texts from file (one per line)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Start interactive mode')
    parser.add_argument('--output', '-o', metavar='FILE',
                       help='Save results to file (JSON format)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed analysis')
    parser.add_argument('--train', action='store_true',
                       help='Train models (run this first time)')
    
    args = parser.parse_args()
    
    # Initialize models
    print("Initializing models...")
    sentiment_analyzer = SentimentAnalyzer()
    text_classifier = TextClassifier()
    
    # Load or train models
    if args.train:
        print("Training models...")
        sentiment_analyzer.train_naive_bayes()
        text_classifier.train_models()
        print("Models trained successfully!")
        return
    
    # Try to load existing models
    if not sentiment_analyzer.load_model():
        print("No sentiment model found. Training...")
        sentiment_analyzer.train_naive_bayes()
    
    if not text_classifier.load_model():
        print("No classifier model found. Training...")
        text_classifier.train_models()
    
    results = []
    
    if args.interactive:
        results = interactive_mode(sentiment_analyzer, text_classifier)
    elif args.batch:
        results = batch_mode(args.batch, sentiment_analyzer, text_classifier, args.classify)
    elif args.text:
        result = analyze_single(args.text, sentiment_analyzer, text_classifier, args.classify, args.verbose)
        results = [result]
    else:
        parser.print_help()
        return
    
    # Save results if requested
    if args.output and results:
        save_results(results, args.output)
        print(f"\nResults saved to {args.output}")

def analyze_single(text, sentiment_analyzer, text_classifier, classify=False, verbose=False):
    """Analyze a single text"""
    print(f"\nAnalyzing: {text}")
    print("=" * 50)
    
    if classify:
        result = text_classifier.predict(text)
        print(f"Category: {result['predicted_category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        if verbose:
            print("\nAll Probabilities:")
            for category, prob in sorted(result['all_probabilities'].items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"  {category}: {prob:.3f}")
    else:
        result = sentiment_analyzer.comprehensive_analysis(text)
        print(f"Sentiment: {result['final_sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        if verbose:
            print("\nDetailed Breakdown:")
            print(f"  VADER: {result['details']['vader']['sentiment']} ({result['details']['vader']['compound']:.3f})")
            print(f"  TextBlob: {result['details']['textblob']['sentiment']} ({result['details']['textblob']['polarity']:.3f})")
            print(f"  Rule-based: {result['details']['rule_based']['sentiment']}")
            print(f"  Naive Bayes: {result['details']['naive_bayes']['sentiment']}")
            
            print("\nVoting Results:")
            for sentiment, count in result['voting_results'].items():
                print(f"  {sentiment}: {count}/4 votes")
    
    return result

def batch_mode(filename, sentiment_analyzer, text_classifier, classify=False):
    """Process texts from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if not texts:
            print("No texts found in file.")
            return []
        
        print(f"\nProcessing {len(texts)} texts from {filename}")
        print("=" * 50)
        
        results = []
        for i, text in enumerate(texts, 1):
            print(f"\n[{i}/{len(texts)}] Processing: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            if classify:
                result = text_classifier.predict(text)
                print(f"  Category: {result['predicted_category']} (Confidence: {result['confidence']:.3f})")
            else:
                result = sentiment_analyzer.comprehensive_analysis(text)
                print(f"  Sentiment: {result['final_sentiment']} (Confidence: {result['confidence']:.3f})")
            
            results.append(result)
        
        # Summary statistics
        if classify:
            categories = {}
            for result in results:
                cat = result['predicted_category']
                categories[cat] = categories.get(cat, 0) + 1
            
            print(f"\nClassification Summary:")
            print("=" * 30)
            for category, count in sorted(categories.items()):
                print(f"{category}: {count}")
        else:
            sentiments = {}
            confidences = []
            for result in results:
                sent = result['final_sentiment']
                sentiments[sent] = sentiments.get(sent, 0) + 1
                confidences.append(result['confidence'])
            
            print(f"\nSentiment Summary:")
            print("=" * 25)
            for sentiment, count in sorted(sentiments.items()):
                percentage = (count / len(results)) * 100
                print(f"{sentiment}: {count} ({percentage:.1f}%)")
            
            avg_confidence = sum(confidences) / len(confidences)
            print(f"\nAverage Confidence: {avg_confidence:.3f}")
        
        return results
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

def interactive_mode(sentiment_analyzer, text_classifier):
    """Interactive mode for continuous analysis"""
    print("\n" + "="*60)
    print("INTERACTIVE SENTIMENT ANALYSIS & TEXT CLASSIFICATION")
    print("="*60)
    print("Commands:")
    print("  'classify <text>' - Classify text into categories")
    print("  'sentiment <text>' - Analyze sentiment")
    print("  '<text>' - Default sentiment analysis")
    print("  'quit' or 'exit' - Exit program")
    print("="*60)
    
    results = []
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Parse command
            if user_input.startswith('classify '):
                text = user_input[9:]
                result = analyze_single(text, sentiment_analyzer, text_classifier, classify=True, verbose=True)
                results.append(result)
            elif user_input.startswith('sentiment '):
                text = user_input[10:]
                result = analyze_single(text, sentiment_analyzer, text_classifier, classify=False, verbose=True)
                results.append(result)
            else:
                # Default to sentiment analysis
                result = analyze_single(user_input, sentiment_analyzer, text_classifier, classify=False, verbose=True)
                results.append(result)
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\nProcessed {len(results)} texts in this session.")
    return results

def save_results(results, filename):
    """Save results to JSON file"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'total_results': len(results),
        'results': results
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()