#!/bin/bash

# Quick Start Script for Agentic AI Blueprint Analyzer
# This script sets up and runs the application locally

set -e

echo "🚀 Agentic AI Blueprint Analyzer - Quick Start"
echo "=============================================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+"
    exit 1
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 20+"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Backend setup
echo "📦 Setting up backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing backend dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  No .env file found in backend directory"
    echo "   Copying .env.example to .env"
    cp .env.example .env
    echo ""
    echo "   ⚠️  IMPORTANT: Edit backend/.env with your Azure OpenAI credentials"
    echo ""
fi

# Start backend in background
echo "Starting backend server..."
python main.py &
BACKEND_PID=$!
echo "Backend running with PID: $BACKEND_PID"

# Frontend setup
cd ../frontend
echo ""
echo "📦 Setting up frontend..."

# Install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Starting frontend development server..."
echo ""
echo "=============================================="
echo "📍 Frontend: http://localhost:5173"
echo "📍 Backend:  http://localhost:8000"
echo "📍 API Docs: http://localhost:8000/docs"
echo "=============================================="
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null || true
    exit 0
}

# Set up trap to cleanup on script exit
trap cleanup INT TERM

# Start frontend (this will block)
npm run dev

# Cleanup will be called when npm run dev is interrupted