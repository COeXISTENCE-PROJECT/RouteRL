#!/bin/bash

cd /app/
echo "--- Installing dependencies ---"
pip3 install --no-cache-dir -r requirements.txt
echo "--- Running main.py ---"
python experiments/trials_on_two_routes/mutation_sac.py