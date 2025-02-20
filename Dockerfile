# hash:sha256:a555d40e3c0f6d8aeaff7e4370f1e7107309b4769bf4352702f1ef769f32717b
FROM registry.codeocean.com/codeocean/mambaforge3:22.11.1-4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV SUMO_HOME=/usr/local/share/sumo
ENV PATH="${PATH}:/usr/local/bin"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake=3.22.1-1ubuntu1.22.04.2 \
        g++=4:11.2.0-1ubuntu1 \
        libfox-1.6-dev=1.6.57-1build1 \
        libgdal-dev=3.4.1+dfsg-1build4 \
        libproj-dev=8.2.1-1 \
        libxerces-c-dev=3.2.3+debian-3ubuntu0.1 \
        python3=3.10.6-1~22.04.1 \
        python3-dev=3.10.6-1~22.04.1 \
        python3-pip=22.0.2+dfsg-1ubuntu0.5 \
        python3-setuptools=59.6.0-1.2ubuntu0.22.04.2 \
        python3-wheel=0.37.1-2ubuntu0.22.04.1 \
        wget=1.21.2-2ubuntu1.1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    routerl==1.0.0

COPY postInstall /
RUN /postInstall
