# Base image Docker source file here:
# https://github.com/intel/ai-containers/blob/main/pytorch/Dockerfile
FROM intel/intel-extension-for-pytorch:2.6.10-xpu

# Install essential dev tools
RUN apt-get update && apt-get install -y \
    openssh-server \
    git \
    build-essential \
    ninja-build \
    cmake \
    neovim \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH for VS Code
RUN mkdir /var/run/sshd \
    && echo 'root:password' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Environment setup
ENV WORKSPACE=/workspace
ENV PYTHONPATH=${WORKSPACE}/src:${WORKSPACE}/src/bnb:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/opt/conda/envs/idp/lib:${LD_LIBRARY_PATH}

# Create workspace structure
RUN mkdir -p ${WORKSPACE}/src
WORKDIR ${WORKSPACE}

# First install packages that don't depend on PyTorch to avoid conflicts
RUN pip install --no-cache-dir \
    ipython \
    pytest \
    ipdb \
    black \
    ruff

# Then install PyTorch-dependent packages with constraint to use existing torch
RUN pip install transformers accelerate bitsandbytes

# Copy the bitsandbytes-intel repository into /workspace/src/bnb and install it.
COPY .. ${WORKSPACE}/src/bnb
RUN cd ${WORKSPACE}/src/bnb && pip install .

COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["sleep", "infinity"]