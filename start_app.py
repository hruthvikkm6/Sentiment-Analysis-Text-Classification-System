#!/usr/bin/env python3
"""
Quick start script for the Sentiment Analysis & Text Classification System
This script handles initial setup and starts the application
"""

import os
import sys
import subprocess
import time

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    return True

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def setup_nltk_data():
    """Download required NLTK data"""
    print("Setting up NLTK data...")
    try:
        subprocess.check_call([sys.executable, "setup_nltk.py"])
        print("✓ NLTK data setup complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to setup NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "static/images",
        "data",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Directories created successfully!")

def run_tests():
    """Run basic tests to ensure everything is working"""
    print("Running system tests...")
    try:
        result = subprocess.run([sys.executable, "run_tests.py"], 
                              capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✓ All tests passed!")
            return True
        else:
            print(f"✗ Some tests failed:")
            print(result.stdout)
            return False
    except subprocess.TimeoutExpired:
        print("⚠ Tests took too long, skipping...")
        return True
    except Exception as e:
        print(f"⚠ Could not run tests: {e}")
        return True

def start_application():
    """Start the Flask application"""
    print("\n" + "="*60)
    print("🚀 STARTING SENTIMENT ANALYSIS SYSTEM")
    print("="*60)
    print("The application will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the application")
    print("="*60)
    
    try:
        # Start the Flask app
        subprocess.call([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except Exception as e:
        print(f"\nError starting application: {e}")

def main():
    """Main setup and start function"""
    print("🧠 Sentiment Analysis & Text Classification System")
    print("=" * 55)
    print("Initializing application...")
    print()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not os.path.exists("venv") and "--skip-install" not in sys.argv:
        install_requirements()
    
    # Setup NLTK data
    setup_nltk_data()
    
    # Run tests (optional)
    if "--with-tests" in sys.argv:
        run_tests()
    
    # Ask user if they want to start the web app
    if "--no-web" not in sys.argv:
        print("\nSetup complete! 🎉")
        print("\nOptions:")
        print("1. Start web application (recommended)")
        print("2. Use command line interface")
        print("3. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == "1":
                    start_application()
                    break
                elif choice == "2":
                    print("\nTo use the command line interface:")
                    print("  python sentiment_analyzer.py \"Your text here\"")
                    print("  python sentiment_analyzer.py --interactive")
                    print("  python sentiment_analyzer.py --help")
                    break
                elif choice == "3":
                    print("Goodbye! 👋")
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)