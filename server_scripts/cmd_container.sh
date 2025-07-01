#!/bin/bash

cd /app/
echo "--- Installing dependencies ---"
pip3 install --no-cache-dir -r requirements.txt
echo "--- Running main.py ---"
python -u tutorials/two_route_net/mappo_marginal.py
