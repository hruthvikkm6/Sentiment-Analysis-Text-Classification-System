# Installation Guide

## Quick Start (Recommended)

1. **Clone or download the project**
2. **Run the setup script:**
   ```bash
   python start_app.py
   ```
   
   This will automatically:
   - Install all dependencies
   - Set up NLTK data
   - Create necessary directories
   - Start the web application

3. **Open your browser** to `http://localhost:5000`

## Manual Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection (for downloading NLTK data and packages)

### Step-by-Step Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK Data**
   ```bash
   python setup_nltk.py
   ```

3. **Create Directories**
   ```bash
   mkdir -p models static/images data logs
   ```

4. **Run Tests (Optional)**
   ```bash
   python run_tests.py
   ```

5. **Start the Application**
   ```bash
   python app.py
   ```

## Virtual Environment (Recommended)

To avoid conflicts with other Python projects:

```bash
# Create virtual environment
python -m venv sentiment_env

# Activate virtual environment
# On Windows:
sentiment_env\Scripts\activate
# On macOS/Linux:
source sentiment_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup NLTK data
python setup_nltk.py

# Start application
python app.py
```

## Usage Options

### 1. Web Interface (Recommended)
```bash
python app.py
# Open http://localhost:5000 in your browser
```

### 2. Command Line Interface
```bash
# Analyze single text
python sentiment_analyzer.py "I love this product!"

# Interactive mode
python sentiment_analyzer.py --interactive

# Classify text
python sentiment_analyzer.py --classify "New AI breakthrough"

# Batch processing
python sentiment_analyzer.py --batch texts.txt

# Show help
python sentiment_analyzer.py --help
```

### 3. Train Models (First time or to retrain)
```bash
python sentiment_analyzer.py --train
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'XXX'**
   - Solution: Install missing packages
   ```bash
   pip install -r requirements.txt
   ```

2. **NLTK Data Error**
   - Solution: Download NLTK data
   ```bash
   python setup_nltk.py
   ```

3. **Permission Denied Errors**
   - Solution: Use virtual environment or run with appropriate permissions

4. **Port 5000 Already in Use**
   - Solution: Stop other applications using port 5000, or modify `app.py` to use a different port:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

5. **Slow Initial Startup**
   - The first run may be slow as models are being trained
   - Subsequent runs will be much faster as models are saved

### Getting Help

1. **Check the logs** for error messages
2. **Run tests** to identify issues:
   ```bash
   python run_tests.py
   ```
3. **Verify installation**:
   ```bash
   python -c "import nltk, flask, pandas, numpy; print('All imports successful!')"
   ```

## System Requirements

### Minimum Requirements
- **RAM**: 4 GB
- **Storage**: 1 GB free space
- **CPU**: Any modern processor

### Recommended Requirements
- **RAM**: 8 GB or more
- **Storage**: 2 GB free space
- **CPU**: Multi-core processor for better performance

## Development Setup

For developers who want to modify the code:

1. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest flake8 black
   ```

2. **Run tests**:
   ```bash
   python run_tests.py
   pytest  # If you have pytest installed
   ```

3. **Code formatting**:
   ```bash
   black .  # Format all Python files
   flake8 . # Check for style issues
   ```

## Docker Setup (Advanced)

If you prefer using Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python setup_nltk.py

EXPOSE 5000
CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t sentiment-analyzer .
docker run -p 5000:5000 sentiment-analyzer
```

## Performance Optimization

1. **Use SSD storage** for better model loading times
2. **Increase RAM** for processing larger datasets
3. **Use caching** for frequently analyzed texts
4. **Consider GPU acceleration** for large-scale deployment (requires additional setup)

## Next Steps

After installation:
1. **Explore the web interface** at `http://localhost:5000`
2. **Try the sample texts** provided in each analysis page
3. **Test batch processing** with your own data
4. **Check the analytics dashboard** for insights
5. **Export results** in various formats

For more advanced usage, see the main README.md file.