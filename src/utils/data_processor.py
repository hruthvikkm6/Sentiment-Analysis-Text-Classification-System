#!/usr/bin/env python3
"""
Data Processing Utilities
Handle data loading, preprocessing, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import json
import csv
from datetime import datetime
import os

class DataProcessor:
    """Data processing and visualization utilities"""
    
    def __init__(self):
        self.sentiment_colors = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#4682B4'    # Steel Blue
        }
    
    def load_csv_data(self, file_path):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    
    def save_results_to_csv(self, results, filename):
        """Save analysis results to CSV"""
        if not results:
            return False
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            if 'error' not in result:
                row = {
                    'text': result['text'],
                    'sentiment': result['final_sentiment'],
                    'confidence': result['confidence'],
                    'vader_sentiment': result['details']['vader']['sentiment'],
                    'vader_compound': result['details']['vader']['compound'],
                    'textblob_sentiment': result['details']['textblob']['sentiment'],
                    'textblob_polarity': result['details']['textblob']['polarity'],
                    'rule_based_sentiment': result['details']['rule_based']['sentiment'],
                    'timestamp': datetime.now().isoformat()
                }
                csv_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        return True
    
    def save_results_to_json(self, results, filename):
        """Save analysis results to JSON"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False
    
    def create_sentiment_distribution_plot(self, results):
        """Create sentiment distribution visualization"""
        if not results:
            return None
        
        # Count sentiments
        sentiments = [r['final_sentiment'] for r in results if 'error' not in r]
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        # Create plotly pie chart
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.3,
            marker_colors=[self.sentiment_colors.get(s, '#808080') for s in sentiment_counts.index]
        )])
        
        fig.update_layout(
            title="Sentiment Distribution",
            font=dict(size=14),
            showlegend=True
        )
        
        return fig
    
    def create_confidence_histogram(self, results):
        """Create confidence score histogram"""
        if not results:
            return None
        
        confidences = [r['confidence'] for r in results if 'error' not in r]
        
        fig = go.Figure(data=[go.Histogram(
            x=confidences,
            nbinsx=20,
            marker_color='skyblue',
            opacity=0.7
        )])
        
        fig.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            font=dict(size=14)
        )
        
        return fig
    
    def create_sentiment_timeline(self, results):
        """Create sentiment timeline if timestamps are available"""
        if not results:
            return None
        
        # Create sample timeline data
        df_timeline = pd.DataFrame([
            {
                'timestamp': datetime.now().timestamp() + i,
                'sentiment': r['final_sentiment'],
                'confidence': r['confidence']
            }
            for i, r in enumerate(results) if 'error' not in r
        ])
        
        df_timeline['datetime'] = pd.to_datetime(df_timeline['timestamp'], unit='s')
        
        fig = px.scatter(
            df_timeline, 
            x='datetime', 
            y='confidence',
            color='sentiment',
            color_discrete_map=self.sentiment_colors,
            title="Sentiment Analysis Timeline"
        )
        
        return fig
    
    def create_wordcloud(self, texts, sentiment_filter=None):
        """Create word cloud from texts"""
        if sentiment_filter:
            # Filter texts by sentiment (would need sentiment results)
            pass
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis'
        ).generate(combined_text)
        
        # Create matplotlib figure
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        
        # Save to file
        if not os.path.exists('static/images'):
            os.makedirs('static/images')
        
        plt.savefig('static/images/wordcloud.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        return 'static/images/wordcloud.png'
    
    def create_comparison_chart(self, results):
        """Create comparison chart of different sentiment analysis methods"""
        if not results:
            return None
        
        methods = ['vader', 'textblob', 'rule_based', 'naive_bayes']
        sentiment_data = {method: [] for method in methods}
        
        for result in results:
            if 'error' not in result:
                for method in methods:
                    if method in result['details']:
                        sentiment_data[method].append(result['details'][method]['sentiment'])
                    else:
                        sentiment_data[method].append('neutral')
        
        # Count sentiments for each method
        method_counts = {}
        for method, sentiments in sentiment_data.items():
            counts = pd.Series(sentiments).value_counts()
            method_counts[method] = counts
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=methods,
            specs=[[{"type": "pie"}, {"type": "pie"}],
                   [{"type": "pie"}, {"type": "pie"}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (method, counts) in enumerate(method_counts.items()):
            row, col = positions[i]
            
            fig.add_trace(
                go.Pie(
                    labels=counts.index,
                    values=counts.values,
                    name=method,
                    marker_colors=[self.sentiment_colors.get(s, '#808080') for s in counts.index]
                ),
                row=row, col=col
            )
        
        fig.update_layout(title_text="Sentiment Analysis Methods Comparison")
        
        return fig
    
    def generate_statistics(self, results):
        """Generate statistical summary of results"""
        if not results:
            return {}
        
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {}
        
        sentiments = [r['final_sentiment'] for r in valid_results]
        confidences = [r['confidence'] for r in valid_results]
        
        stats = {
            'total_analyzed': len(valid_results),
            'sentiment_distribution': pd.Series(sentiments).value_counts().to_dict(),
            'confidence_stats': {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'positive_percentage': (sentiments.count('positive') / len(sentiments)) * 100,
            'negative_percentage': (sentiments.count('negative') / len(sentiments)) * 100,
            'neutral_percentage': (sentiments.count('neutral') / len(sentiments)) * 100
        }
        
        return stats
    
    def export_to_pdf_report(self, results, filename="sentiment_analysis_report.pdf"):
        """Generate PDF report (placeholder - would need reportlab)"""
        # This would require additional PDF generation library
        print(f"PDF report functionality would generate: {filename}")
        return filename

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_text_input(text):
        """Validate text input"""
        if not text or not isinstance(text, str):
            return False, "Text must be a non-empty string"
        
        if len(text.strip()) == 0:
            return False, "Text cannot be empty or only whitespace"
        
        if len(text) > 10000:
            return False, "Text is too long (max 10,000 characters)"
        
        return True, "Valid"
    
    @staticmethod
    def validate_batch_input(texts):
        """Validate batch text input"""
        if not texts or not isinstance(texts, list):
            return False, "Input must be a list of texts"
        
        if len(texts) == 0:
            return False, "List cannot be empty"
        
        if len(texts) > 100:
            return False, "Too many texts (max 100 per batch)"
        
        for i, text in enumerate(texts):
            valid, message = DataValidator.validate_text_input(text)
            if not valid:
                return False, f"Text {i+1}: {message}"
        
        return True, "Valid"

def main():
    """Test data processing utilities"""
    processor = DataProcessor()
    
    # Sample results for testing
    sample_results = [
        {
            'text': 'I love this product!',
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
            'text': 'This is terrible!',
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
    
    # Test statistics generation
    stats = processor.generate_statistics(sample_results)
    print("Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Test data saving
    processor.save_results_to_csv(sample_results, 'test_results.csv')
    processor.save_results_to_json(sample_results, 'test_results.json')
    
    print("Data processing utilities tested successfully!")

if __name__ == "__main__":
    main()