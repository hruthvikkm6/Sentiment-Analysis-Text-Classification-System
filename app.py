#!/usr/bin/env python3
"""
Flask Web Application for Sentiment Analysis & Text Classification System
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import json
from datetime import datetime
import traceback

# Add src to path
sys.path.append('src')

from models.sentiment_analyzer import SentimentAnalyzer
from models.text_classifier import TextClassifier
from utils.data_processor import DataProcessor, DataValidator

app = Flask(__name__)
CORS(app)

# Initialize models
sentiment_analyzer = SentimentAnalyzer()
text_classifier = TextClassifier()
data_processor = DataProcessor()

# Global variables to store analysis history
analysis_history = []

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/sentiment')
def sentiment_page():
    """Sentiment analysis page"""
    return render_template('sentiment.html')

@app.route('/classification')
def classification_page():
    """Text classification page"""
    return render_template('classification.html')

@app.route('/batch')
def batch_page():
    """Batch analysis page"""
    return render_template('batch.html')

@app.route('/analytics')
def analytics_page():
    """Analytics dashboard page"""
    return render_template('analytics.html')

# API Routes

@app.route('/api/analyze', methods=['POST'])
def api_analyze_sentiment():
    """API endpoint for sentiment analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Validate input
        valid, message = DataValidator.validate_text_input(text)
        if not valid:
            return jsonify({'error': message}), 400
        
        # Analyze sentiment
        result = sentiment_analyzer.comprehensive_analysis(text)
        
        # Add to history
        result['timestamp'] = datetime.now().isoformat()
        analysis_history.append(result)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/classify', methods=['POST'])
def api_classify_text():
    """API endpoint for text classification"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Validate input
        valid, message = DataValidator.validate_text_input(text)
        if not valid:
            return jsonify({'error': message}), 400
        
        # Classify text
        result = text_classifier.predict(text)
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

@app.route('/api/batch', methods=['POST'])
def api_batch_analysis():
    """API endpoint for batch analysis"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        analysis_type = data.get('type', 'sentiment')  # 'sentiment' or 'classification'
        
        # Validate input
        valid, message = DataValidator.validate_batch_input(texts)
        if not valid:
            return jsonify({'error': message}), 400
        
        if analysis_type == 'sentiment':
            results = sentiment_analyzer.batch_analysis(texts)
            # Add to history
            for result in results:
                result['timestamp'] = datetime.now().isoformat()
                analysis_history.append(result)
        elif analysis_type == 'classification':
            results = text_classifier.batch_predict(texts)
            for result in results:
                result['timestamp'] = datetime.now().isoformat()
        else:
            return jsonify({'error': 'Invalid analysis type'}), 400
        
        return jsonify({'results': results, 'count': len(results)})
        
    except Exception as e:
        return jsonify({'error': f'Batch analysis failed: {str(e)}'}), 500

@app.route('/api/statistics')
def api_get_statistics():
    """API endpoint to get analysis statistics"""
    try:
        stats = data_processor.generate_statistics(analysis_history)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'Failed to generate statistics: {str(e)}'}), 500

@app.route('/api/export/<format>')
def api_export_data(format):
    """API endpoint to export analysis results"""
    try:
        if not analysis_history:
            return jsonify({'error': 'No data to export'}), 400
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            filename = f'sentiment_analysis_{timestamp}.csv'
            success = data_processor.save_results_to_csv(analysis_history, filename)
            if success:
                return send_file(filename, as_attachment=True)
        elif format == 'json':
            filename = f'sentiment_analysis_{timestamp}.json'
            success = data_processor.save_results_to_json(analysis_history, filename)
            if success:
                return send_file(filename, as_attachment=True)
        else:
            return jsonify({'error': 'Invalid format. Use csv or json'}), 400
        
        return jsonify({'error': 'Export failed'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/api/charts/sentiment_distribution')
def api_sentiment_distribution():
    """API endpoint for sentiment distribution chart"""
    try:
        fig = data_processor.create_sentiment_distribution_plot(analysis_history)
        if fig:
            return jsonify(fig.to_dict())
        return jsonify({'error': 'No data available'}), 400
    except Exception as e:
        return jsonify({'error': f'Chart generation failed: {str(e)}'}), 500

@app.route('/api/charts/confidence_histogram')
def api_confidence_histogram():
    """API endpoint for confidence histogram"""
    try:
        fig = data_processor.create_confidence_histogram(analysis_history)
        if fig:
            return jsonify(fig.to_dict())
        return jsonify({'error': 'No data available'}), 400
    except Exception as e:
        return jsonify({'error': f'Chart generation failed: {str(e)}'}), 500

@app.route('/api/charts/methods_comparison')
def api_methods_comparison():
    """API endpoint for methods comparison chart"""
    try:
        fig = data_processor.create_comparison_chart(analysis_history)
        if fig:
            return jsonify(fig.to_dict())
        return jsonify({'error': 'No data available'}), 400
    except Exception as e:
        return jsonify({'error': f'Chart generation failed: {str(e)}'}), 500

@app.route('/api/wordcloud')
def api_generate_wordcloud():
    """API endpoint to generate word cloud"""
    try:
        if not analysis_history:
            return jsonify({'error': 'No data available'}), 400
        
        texts = [result['text'] for result in analysis_history if 'error' not in result]
        image_path = data_processor.create_wordcloud(texts)
        
        return jsonify({'image_path': image_path})
        
    except Exception as e:
        return jsonify({'error': f'Word cloud generation failed: {str(e)}'}), 500

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'sentiment_model_trained': sentiment_analyzer.model_trained,
        'classifier_model_trained': text_classifier.trained,
        'analysis_history_count': len(analysis_history),
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def initialize_models():
    """Initialize and train models if needed"""
    print("Initializing models...")
    
    # Try to load existing models
    sentiment_loaded = sentiment_analyzer.load_model()
    classifier_loaded = text_classifier.load_model()
    
    # Train models if not loaded
    if not sentiment_loaded:
        print("Training sentiment analysis model...")
        sentiment_analyzer.train_naive_bayes()
    
    if not classifier_loaded:
        print("Training text classifier model...")
        text_classifier.train_models()
    
    print("Models initialized successfully!")

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize models
    initialize_models()
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)