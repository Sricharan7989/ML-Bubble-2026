#!/usr/bin/env bash
# One-command reproduction of the full experiment.
set -e
pip install -r requirements.txt
python main.py
echo "Done. See ./outputs for metrics + figures, ./models for saved models."
echo "Serve API : uvicorn app.api:app --port 8000"
echo "Dashboard : streamlit run app/dashboard.py"
