#!/usr/bin/env python3
"""
Test script to demonstrate model training and compatibility
"""

import os
import sys
from train_sentiment_model import SentimentModelTrainer
from train_text_classifier import TextClassifierTrainer

def test_sentiment_training():
    """Test sentiment model training with sample dataset"""
    print("="*60)
    print("TESTING SENTIMENT MODEL TRAINING")
    print("="*60)
    
    trainer = SentimentModelTrainer()
    
    # Train with sample dataset
    results = trainer.train_model(
        dataset_path='sample_sentiment_dataset.csv',
        text_column='text',
        label_column='sentiment'
    )
    
    if results:
        print(f"\n✓ Sentiment model training completed successfully!")
        print(f"✓ Accuracy: {results['accuracy']:.4f}")
        
        # Test compatibility
        if trainer.test_model_compatibility():
            print("✓ Model compatibility verified!")
        return True
    else:
        print("✗ Sentiment model training failed!")
        return False

def test_text_classification_training():
    """Test text classification model training with sample dataset"""
    print("\n" + "="*60)
    print("TESTING TEXT CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    trainer = TextClassifierTrainer()
    
    # Train with sample dataset
    results = trainer.train_models(
        dataset_path='sample_text_classification_dataset.csv',
        text_column='text',
        label_column='category'
    )
    
    if results and trainer.best_model:
        print(f"\n✓ Text classification training completed successfully!")
        print(f"✓ Best model: {trainer.best_model_name}")
        
        # Find best accuracy
        best_acc = max([r.get('accuracy', 0) for r in results.values() if isinstance(r, dict) and 'accuracy' in r])
        print(f"✓ Best accuracy: {best_acc:.4f}")
        
        # Test compatibility
        if trainer.test_model_compatibility():
            print("✓ Model compatibility verified!")
        return True
    else:
        print("✗ Text classification training failed!")
        return False

def test_existing_system_compatibility():
    """Test if the trained models work with the existing system"""
    print("\n" + "="*60)
    print("TESTING EXISTING SYSTEM COMPATIBILITY")
    print("="*60)
    
    try:
        # Test with existing sentiment analyzer
        sys.path.append('src/models')
        from sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        if analyzer.load_model():
            test_text = "This is a fantastic product with amazing quality!"
            result = analyzer.comprehensive_analysis(test_text)
            print(f"✓ Sentiment Analysis Test:")
            print(f"  Text: {test_text}")
            print(f"  Result: {result['final_sentiment']} (confidence: {result['confidence']:.3f})")
        else:
            print("✗ Failed to load sentiment model")
            return False
        
        # Test with existing text classifier
        from text_classifier import TextClassifier
        
        classifier = TextClassifier()
        if classifier.load_model():
            test_text = "New smartphone features advanced AI capabilities and improved camera"
            result = classifier.predict(test_text)
            print(f"\n✓ Text Classification Test:")
            print(f"  Text: {test_text}")
            print(f"  Category: {result['predicted_category']} (confidence: {result['confidence']:.3f})")
        else:
            print("✗ Failed to load text classification model")
            return False
        
        print(f"\n✓ All compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        return False

def main():
    """Run all training tests"""
    print("Starting Model Training Tests...")
    print("This will train models using the sample datasets and test compatibility.")
    print("\nNote: This will overwrite any existing model files in the models/ directory.")
    
    # Ask for confirmation
    response = input("\nDo you want to continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Training tests cancelled.")
        return
    
    all_passed = True
    
    # Test sentiment model training
    if not test_sentiment_training():
        all_passed = False
    
    # Test text classification training
    if not test_text_classification_training():
        all_passed = False
    
    # Test compatibility with existing system
    if not test_existing_system_compatibility():
        all_passed = False
    
    # Final results
    print("\n" + "="*60)
    print("TRAINING TEST RESULTS")
    print("="*60)
    
    if all_passed:
        print("✓ All tests passed successfully!")
        print("✓ Models are trained and compatible with existing system")
        print("\nYou can now use your trained models with:")
        print("  - python app.py (web application)")
        print("  - python demo.py (demo script)")
        print("  - src/models/sentiment_analyzer.py")
        print("  - src/models/text_classifier.py")
    else:
        print("✗ Some tests failed. Check the output above for details.")
        print("✗ Please review the error messages and try again.")
    
    print(f"\nModel files saved in: {os.path.abspath('models/')}")

if __name__ == "__main__":
    main()