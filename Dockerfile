FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y requirements.txt && conda clean -ya