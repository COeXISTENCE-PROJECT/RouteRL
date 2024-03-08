#!/bin/bash

cd /app/
echo "--- Installing dependencies ---"
pip install --no-cache-dir -r requirements.txt
echo "--- Running main.py ---"
python main.py