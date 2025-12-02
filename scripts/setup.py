#!/usr/bin/env python3
"""
Setup script for Soccer Prediction AI System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("Python 3.9 or higher is required")
        sys.exit(1)
    print(f"Python {version.major}.{version.minor}.{version.micro} - OK")

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"Requirements file {requirements_file} not found")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "logs",
        "output/predictions",
        "output/reports",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("Directories created successfully")

def setup_database():
    """Setup database"""
    print("\nDatabase Setup")
    print("="*50)
    
    # Check if PostgreSQL is installed
    try:
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("PostgreSQL found:", result.stdout.strip())
        else:
            print("PostgreSQL not found. Please install PostgreSQL:")
            print("  Windows: https://www.postgresql.org/download/windows/")
            print("  macOS: brew install postgresql")
            print("  Linux: sudo apt-get install postgresql")
    except FileNotFoundError:
        print("PostgreSQL not found. Please install PostgreSQL")
    
    print("\nYou can also use SQLite by modifying config/config.yaml")

def setup_config():
    """Create configuration files"""
    config_dir = Path("config")
    
    # Create config.yaml if not exists
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        sample_config = """# Soccer Prediction AI Configuration

system:
  device: "cpu"
  batch_size: 32
  num_workers: 4
  cache_dir: "./data/cache"

data_collection:
  football_data_api_key: "YOUR_API_KEY_HERE"
  api_football_key: "YOUR_API_KEY_HERE"
  weather_api_key: "YOUR_API_KEY_HERE"

models:
  ensemble_weights:
    xgboost: 0.35
    catboost: 0.30
    neural_network: 0.25
    gradient_boosting: 0.10

prediction:
  min_confidence: 0.65
  max_predictions_per_day: 50

database:
  type: "sqlite"
  path: "./data/soccer_predictions.db"
"""
        config_file.write_text(sample_config)
        print(f"Created sample config file: {config_file}")
    
    # Create API keys template
    api_keys_file = config_dir / "api_keys.json"
    if not api_keys_file.exists():
        sample_keys = {
            "football_data": "YOUR_API_KEY_HERE",
            "api_football": "YOUR_API_KEY_HERE",
            "openweather": "YOUR_API_KEY_HERE"
        }
        import json
        api_keys_file.write_text(json.dumps(sample_keys, indent=2))
        print(f"Created API keys template: {api_keys_file}")

def setup_chromedriver():
    """Setup ChromeDriver for web scraping"""
    system = platform.system()
    
    print(f"\nChromeDriver Setup for {system}")
    print("="*50)
    
    if system == "Windows":
        print("Please download ChromeDriver from: https://chromedriver.chromium.org/")
        print("Extract chromedriver.exe to the scripts/ directory")
    elif system == "Darwin":  # macOS
        try:
            subprocess.run(['brew', 'install', 'chromedriver'], check=True)
            print("ChromeDriver installed via Homebrew")
        except:
            print("Install via Homebrew or download from: https://chromedriver.chromium.org/")
    elif system == "Linux":
        try:
            subprocess.run(['sudo', 'apt-get', 'install', 'chromium-chromedriver'], check=True)
            print("ChromeDriver installed via apt")
        except:
            print("Install via package manager or download from: https://chromedriver.chromium.org/")

def main():
    """Main setup function"""
    print("="*60)
    print("Soccer Prediction AI System - Setup")
    print("="*60)
    
    # Step 1: Check Python
    check_python_version()
    
    # Step 2: Create directories
    setup_directories()
    
    # Step 3: Install requirements
    if not install_requirements():
        print("Warning: Some requirements may not have installed correctly")
    
    # Step 4: Setup configuration
    setup_config()
    
    # Step 5: Setup database
    setup_database()
    
    # Step 6: Setup ChromeDriver
    setup_chromedriver()
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Edit config/config.yaml with your API keys")
    print("2. Run: python src/main.py --mode=train (to train initial models)")
    print("3. Run: python src/main.py --mode=predict (for daily predictions)")
    print("4. Check output/predictions/ for results")
    print("\nNote: You may need to install additional system dependencies")
    print("      based on your operating system.")

if __name__ == "__main__":
    main()
