FROM python:3.9

LABEL org.opencontainers.image.description="Tripy development container"

WORKDIR /tripy

ENTRYPOINT ["/bin/bash"]

# Setup user account
ARG uid=1000
ARG gid=1000
ENV DEBIAN_FRONTEND=noninteractive

RUN groupadd -r -f -g ${gid} trtuser && \
    useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser && \
    usermod -aG sudo trtuser && \
    echo 'trtuser:nvidia' | chpasswd && \
    mkdir -p /workspace && chown trtuser /workspace

RUN apt-get update && \
    apt-get install -y sudo python3 python3-pip gdb git wget curl && \
    apt-get clean && \
    python3 -m pip install --upgrade pip

COPY .lldbinit /root/
COPY pyproject.toml /tripy/pyproject.toml

RUN pip install build .[docs,dev,test,build] \
    -f https://nvidia.github.io/TensorRT-Incubator/packages.html \
    --extra-index-url https://download.pytorch.org/whl \
    --extra-index-url https://pypi.nvidia.com

# Export tripy into the PYTHONPATH so it doesn't need to be installed after making changes
ENV PYTHONPATH=/tripy
