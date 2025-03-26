#!/bin/bash

cd /app/
echo "--- Installing dependencies ---"
pip3 install --no-cache-dir -r requirements.txt
echo "--- Running main.py ---"
python tutorials/5_CustomDemand/mappo_mutation.py
