FROM --platform=linux/amd64 ubuntu:noble

# Install dependencies
RUN apt update && \
    apt install -y \
    python3-dev \
    python3-venv \
    build-essential \
    jq \
    git \
    curl \
    make \
    clang \
    pkg-config \
    libssl-dev \
    llvm \
    libudev-dev \
    protobuf-compiler \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Install Rust
ENV RUST_TOOLCHAIN=nightly-2024-09-30
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    /root/.cargo/bin/rustup toolchain install ${RUST_TOOLCHAIN} && \
    /root/.cargo/bin/rustup default ${RUST_TOOLCHAIN} && \
    /root/.cargo/bin/rustup toolchain remove stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Jolt
ENV JOLT_VERSION=dd9e5c4bcf36ffeb75a576351807f8d86c33ec66
RUN cargo +${RUST_TOOLCHAIN} install --git https://github.com/a16z/jolt --rev ${JOLT_VERSION} --force --bins jolt

# Install node et al.
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash && \
    export NVM_DIR="/root/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" && \
    nvm install 20 && \
    npm install --prefix /root/.snarkjs snarkjs@0.7.4 && \
    ln -s $(which node) /usr/bin/node && \
    ln -s $(which npm) /usr/bin/npm

# Use a venv because of https://peps.python.org/pep-0668/
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/python3 -m pip install --upgrade pip && \
    rm -rf /root/.cache/pip && \
    echo "source /opt/venv/bin/activate" >> ~/.bashrc
ENV PATH="/opt/venv/bin:${PATH}"

# Install Python dependencies
COPY requirements.txt /opt/omron/requirements.txt
RUN TORCH_VERSION=$(grep "^torch" /opt/omron/requirements.txt) && \
    pip3 install $TORCH_VERSION --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install -r /opt/omron/requirements.txt && \
    rm -rf /root/.cache/pip

# Copy omron
COPY neurons /opt/omron/neurons

# Set workdir for running miner.py or validator.py and compile circuits
WORKDIR /opt/omron/neurons
ENV OMRON_NO_AUTO_UPDATE=1
RUN OMRON_DOCKER_BUILD=1 python3 miner.py && \
    rm -rf /opt/omron/neurons/deployment_layer/*/target/release/build && \
    rm -rf /opt/omron/neurons/deployment_layer/*/target/release/deps && \
    rm -rf /opt/omron/neurons/deployment_layer/*/target/release/examples && \
    rm -rf /opt/omron/neurons/deployment_layer/*/target/release/incremental && \
    rm -rf /root/.bittensor
ENTRYPOINT ["/opt/venv/bin/python3"]
CMD ["-c", "import subprocess; \
    subprocess.run(['/opt/venv/bin/python3', '/opt/omron/neurons/miner.py', '--help']); \
    subprocess.run(['/opt/venv/bin/python3', '/opt/omron/neurons/validator.py', '--help']);" \
    ]
EXPOSE 8091/tcp
EXPOSE 8443/tcp
