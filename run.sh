#!/usr/bin/env bash
# One-command reproduction of the full experiment.
set -e
python -m pip install -r requirements.txt
python main.py
echo "Done. See ./outputs for metrics + figures, ./models for saved models."
echo "Serve API : python -m uvicorn app.api:app --port 8000"
echo "Dashboard : python -m streamlit run app/dashboard.py"
echo "(Windows: use 'py' in place of 'python'.)"
