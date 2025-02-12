FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    sumo sumo-tools sumo-doc \
    && rm -rf /var/lib/apt/lists/* \
    && echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc

# Set SUMO_HOME environment variable
ENV SUMO_HOME=/usr/share/sumo

# Install python requirements
RUN conda install -y requirements.txt && conda clean -ya