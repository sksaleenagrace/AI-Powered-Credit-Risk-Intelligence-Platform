#!/bin/bash

# AI-Powered Credit Risk Intelligence Platform - Setup Script
# This script automates the environment setup and pipeline execution

set -e  # Exit on error

echo "=========================================="
echo "Credit Risk Platform Setup Script"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

echo "✓ Python 3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your GROQ_API_KEY"
    echo "   Get your API key from: https://console.groq.com/keys"
else
    echo "✓ .env file exists"
fi

# Check if GROQ_API_KEY is set
if ! grep -q "GROQ_API_KEY=" .env || grep -q "GROQ_API_KEY=$" .env; then
    echo "⚠️  GROQ_API_KEY not set in .env file"
    echo "   Please add your GROQ_API_KEY to .env file"
    echo "   Get your API key from: https://console.groq.com/keys"
else
    echo "✓ GROQ_API_KEY is set"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data models notebooks documents
echo "✓ Directories created"

# Check if sample data exists
if [ -f "data/sample_data.csv" ]; then
    echo "✓ Sample data found (demo mode ready)"
else
    echo "⚠️  Sample data not found. The application will use demo mode if no full dataset is provided."
fi

# Check if pre-trained model exists
if [ -f "models/lgbm_model.pkl" ]; then
    echo "✓ Pre-trained model found"
else
    echo "⚠️  Pre-trained model not found. You may need to train the model:"
    echo "   python src/train.py"
fi

# Check if pre-generated charts exist
if [ -f "notebooks/shap_summary_plot.png" ]; then
    echo "✓ Pre-generated charts found"
else
    echo "⚠️  Pre-generated charts not found. You may need to run:"
    echo "   python src/explain.py"
    echo "   python src/eda.py"
fi

echo ""
echo "=========================================="
echo "✓ Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. If you haven't already, add your GROQ_API_KEY to .env file"
echo "2. For full dataset, download from Kaggle and place in data/ folder:"
echo "   https://www.kaggle.com/competitions/home-credit-default-risk/data"
echo "3. Run the application:"
echo "   Docker: docker-compose up --build"
echo "   Local: streamlit run ui/app.py"
echo ""
echo "For demo mode (with sample data), just run the application!"
echo ""
