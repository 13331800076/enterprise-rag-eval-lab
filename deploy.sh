#!/bin/bash
set -e

# One-click deployment script for RAG Lab
# Usage: ./deploy.sh [platform]
# Platforms: docker, streamlit, fastapi

PLATFORM=${1:-docker}

echo "🚀 RAG Lab Deployment Script"
echo "Platform: $PLATFORM"

if [ "$PLATFORM" = "docker" ]; then
    echo "Building Docker image..."
    docker build -t rag-lab:latest .
    echo "Starting containers..."
    docker-compose up -d
    echo "✅ RAG Lab is running at http://localhost:8000"
    echo "📊 Streamlit UI at http://localhost:8501"

elif [ "$PLATFORM" = "streamlit" ]; then
    echo "Installing dependencies..."
    pip install -e ".[dev]"
    echo "Starting Streamlit..."
    streamlit run app.py

elif [ "$PLATFORM" = "fastapi" ]; then
    echo "Installing dependencies..."
    pip install -e ".[dev]"
    echo "Starting FastAPI..."
    uvicorn rag_lab.api.app:app --reload

else
    echo "Unknown platform: $PLATFORM"
    echo "Usage: ./deploy.sh [docker|streamlit|fastapi]"
    exit 1
fi
