# This is a Dockerfile for building a Docker image with SUMO 1.19.0 and RouteRL 1.0.0
# It is a template for building an encapsulated environment for running experiments with RouteRL.

FROM --platform=linux/amd64 python:3.12

# Needed tools for building
RUN apt-get update && apt-get install -y \
    wget \
    cmake \
    g++ \
    libxerces-c-dev \
    libproj-dev \
    libfox-1.6-dev \
    libgdal-dev \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Download and build SUMO 1.19.0
RUN wget https://sumo.dlr.de/releases/1.19.0/sumo-src-1.19.0.tar.gz && \
    tar -xzf sumo-src-1.19.0.tar.gz && \
    rm sumo-src-1.19.0.tar.gz
RUN cmake -B build sumo-1.19.0/ && \
    cmake --build build -j$(nproc)
RUN cmake --install build
 
# Set SUMO_HOME
ENV SUMO_HOME=/usr/local/share/sumo
ENV PATH="${PATH}:/usr/local/bin"

# Check SUMO version
RUN echo "$(sumo --version)"

# Install RouteRL (installs with all dependencies)
RUN python3 -m pip install routerl==1.0.0