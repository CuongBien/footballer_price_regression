#!/bin/bash
echo "========================================"
echo "  Footballer Value Prediction App"
echo "========================================"
echo ""
echo "Starting Streamlit app..."
echo ""
echo "App will open at: http://localhost:8501"
echo "Press Ctrl+C to stop the app"
echo ""

source .venv/bin/activate
streamlit run streamlit_app.py
