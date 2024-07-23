#!/bin/bash

cd /app/
echo "--- Installing dependencies ---"
pip3 install --no-cache-dir -r requirements.txt
echo "--- Running main.py ---"
python experiments/mutation-ppo-sac-experiments/mutation_sac.py