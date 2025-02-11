FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.12-ubuntu22.04
WORKDIR /code
ENV SUMO_HOME=/usr/share/sumo
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt
COPY . .