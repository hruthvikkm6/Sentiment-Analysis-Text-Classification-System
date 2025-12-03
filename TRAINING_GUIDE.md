# Model Training Guide

This guide explains how to train custom models from scratch while maintaining compatibility with the existing prediction system.

## Overview

The training scripts create model files that are fully compatible with the existing `SentimentAnalyzer` and `TextClassifier` classes. After training, your custom models will work seamlessly with the web application and demo scripts.

## Files Created

- `train_sentiment_model.py` - Train sentiment analysis models
- `train_text_classifier.py` - Train text classification models
- `sample_sentiment_dataset.csv` - Example sentiment dataset format
- `sample_text_classification_dataset.csv` - Example text classification dataset format

## Quick Start

### 1. Train Sentiment Analysis Model

#### Using Default Dataset (NLTK Movie Reviews)
```bash
python train_sentiment_model.py
```

#### Using Custom Dataset
```bash
python train_sentiment_model.py --dataset your_sentiment_data.csv --text_column "review_text" --label_column "sentiment"
```

### 2. Train Text Classification Model

#### Using Default Dataset (Sample Categories)
```bash
python train_text_classifier.py
```

#### Using Custom Dataset
```bash
python train_text_classifier.py --dataset your_classification_data.csv --text_column "content" --label_column "category"
```

## Dataset Format Requirements

### Sentiment Analysis Dataset

Your CSV file should have these columns:
- **Text column**: Contains the text to analyze
- **Sentiment column**: Contains 'positive' or 'negative' labels

Example:
```csv
text,sentiment
"I love this product!",positive
"This is terrible",negative
"Amazing quality and fast delivery",positive
```

### Text Classification Dataset

Your CSV file should have these columns:
- **Text column**: Contains the text to classify
- **Category column**: Contains the category labels (any string)

Example:
```csv
text,category
"New iPhone released with AI features",Technology
"Basketball team wins championship",Sports
"Election results announced",Politics
```

## Command Line Options

### Sentiment Model Training

```bash
python train_sentiment_model.py [OPTIONS]

Options:
  --dataset PATH          Path to custom CSV dataset
  --text_column STR       Name of text column (default: 'text')
  --label_column STR      Name of sentiment column (default: 'sentiment')
  --test_size FLOAT       Test set proportion (default: 0.2)
  --random_state INT      Random state for reproducibility (default: 42)
  --no_save              Do not save the trained model
```

### Text Classification Training

```bash
python train_text_classifier.py [OPTIONS]

Options:
  --dataset PATH          Path to custom CSV dataset
  --text_column STR       Name of text column (default: 'text')
  --label_column STR      Name of category column (default: 'category')
  --test_size FLOAT       Test set proportion (default: 0.2)
  --random_state INT      Random state for reproducibility (default: 42)
  --no_save              Do not save the trained model
```

## Model Files Generated

After training, these files will be created in the `models/` directory:

### Sentiment Analysis
- `nb_model.pkl` - Trained Naive Bayes model
- `vectorizer.pkl` - TF-IDF vectorizer

### Text Classification
- `text_classifier.pkl` - Best performing model (Naive Bayes, Logistic Regression, or Random Forest)
- `text_vectorizer.pkl` - TF-IDF vectorizer
- `classifier_metadata.pkl` - Model metadata (categories, model type)

## Compatibility

The trained models are fully compatible with:
- `src/models/sentiment_analyzer.py` - Will automatically load your custom sentiment model
- `src/models/text_classifier.py` - Will automatically load your custom classification model
- `app.py` - Web application will use your custom models
- `demo.py` - Demo scripts will use your custom models

## Data Preprocessing

Both training scripts automatically:
1. **Clean text**: Remove URLs, special characters, normalize case
2. **Tokenize**: Split text into words
3. **Remove stopwords**: Filter out common words (the, and, is, etc.)
4. **Lemmatize**: Reduce words to base forms (running → run)
5. **Vectorize**: Convert text to numerical features using TF-IDF

## Model Selection (Text Classification Only)

The text classification trainer evaluates three algorithms:
1. **Naive Bayes**: Fast, works well with small datasets
2. **Logistic Regression**: Balanced performance, interpretable
3. **Random Forest**: Handles complex patterns, robust to overfitting

The best performing model (highest accuracy) is automatically selected and saved.

## Training Tips

### For Better Results:
1. **More Data**: Aim for at least 100+ samples per category
2. **Balanced Data**: Try to have similar amounts of data for each category/sentiment
3. **Quality Data**: Clean, relevant text produces better models
4. **Diverse Data**: Include varied writing styles and contexts

### Troubleshooting:
- **Low Accuracy**: Add more training data or clean existing data
- **Imbalanced Classes**: Collect more samples for underrepresented categories
- **Memory Issues**: Reduce `max_features` in TfidfVectorizer (default: 5000)

## Example Workflow

1. **Prepare your dataset** in CSV format
2. **Train the model**:
   ```bash
   python train_sentiment_model.py --dataset my_data.csv
   ```
3. **Check compatibility** (automatic test runs after training)
4. **Use your model** with existing application:
   ```bash
   python app.py  # Web application
   python demo.py # Demo script
   ```

## Advanced Usage

### Custom Preprocessing
Modify the `preprocess_text()` method in the training scripts to:
- Handle domain-specific text patterns
- Preserve important punctuation or symbols
- Add custom text cleaning steps

### Hyperparameter Tuning
Modify the model parameters in the training scripts:
- Adjust TF-IDF parameters (`max_features`, `ngram_range`)
- Tune model parameters (regularization for Logistic Regression, etc.)
- Change train/test split ratios

### Multiple Model Comparison
The text classification trainer shows performance for all models. Use this information to:
- Choose models based on accuracy vs speed tradeoffs
- Understand which algorithms work best for your data
- Debug performance issues

## Next Steps

After training your models:
1. Test them with the web application (`python app.py`)
2. Evaluate on your own test cases
3. Iterate with more data if needed
4. Deploy to production environment

For questions or issues, check the training output for detailed error messages and suggestions.