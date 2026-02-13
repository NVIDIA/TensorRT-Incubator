FROM python:3.9

LABEL org.opencontainers.image.description="Tripy Development Container"

WORKDIR /tripy

ENTRYPOINT ["/bin/bash"]

# Setup user account
ARG uid=1000
ARG gid=1000
ARG username=trtuser
ENV DEBIAN_FRONTEND=noninteractive

RUN groupadd -r -f -g ${gid} ${username} && \
    useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash ${username} && \
    usermod -aG sudo ${username} && \
    echo "${username}:nvidia" | chpasswd && \
    mkdir -p /workspace && chown ${username} /workspace

RUN apt-get update && \
    apt-get install -y sudo python3 python3-pip gdb git wget curl zsh && \
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
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
