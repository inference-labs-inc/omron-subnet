FROM --platform=linux/amd64 ubuntu:noble

# Install dependencies
RUN apt update && \
    apt install -y \
    pipx \
    build-essential \
    jq \
    git \
    aria2 \
    curl \
    make \
    clang \
    pkg-config \
    libssl-dev \
    llvm \
    libudev-dev \
    protobuf-compiler \
    gosu \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Use ubuntu user
USER ubuntu
WORKDIR /home/ubuntu

# Install Rust
ENV RUST_TOOLCHAIN=nightly-2024-09-30
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    ~/.cargo/bin/rustup toolchain install ${RUST_TOOLCHAIN} && \
    ~/.cargo/bin/rustup default ${RUST_TOOLCHAIN} && \
    ~/.cargo/bin/rustup toolchain remove stable
ENV PATH="~/.cargo/bin:${PATH}"

# Install Jolt
#ENV JOLT_VERSION=dd9e5c4bcf36ffeb75a576351807f8d86c33ec66
#RUN cargo +${RUST_TOOLCHAIN} install --git https://github.com/a16z/jolt --rev ${JOLT_VERSION} --force --bins jolt

# Install node et al.
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash && \
    export NVM_DIR="/home/ubuntu/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" && \
    nvm install 20 && \
    npm install --prefix /home/ubuntu/.snarkjs snarkjs@0.7.4 && \
    mkdir -p ~/.local/bin && \
    ln -s $(which node) /home/ubuntu/.local/bin/node && \
    ln -s $(which npm) /home/ubuntu/.local/bin/npm
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# Copy omron and install Python dependencies (make sure owner is ubuntu)
COPY neurons /home/ubuntu/omron/neurons
COPY pyproject.toml /home/ubuntu/omron/pyproject.toml
COPY uv.lock /home/ubuntu/omron/uv.lock
USER root
RUN chown -R ubuntu:ubuntu /home/ubuntu/omron
USER ubuntu
RUN pipx install uv && \
    cd ~/omron && \
    ~/.local/bin/uv sync --locked --no-dev --compile-bytecode && \
    ~/.local/bin/uv cache clean && \
    echo "source ~/omron/.venv/bin/activate" >> ~/.bashrc
ENV PATH="/home/ubuntu/omron/.venv/bin:${PATH}"

# Set workdir for running miner.py or validator.py and compile circuits
WORKDIR /home/ubuntu/omron/neurons
ENV OMRON_NO_AUTO_UPDATE=1
RUN OMRON_DOCKER_BUILD=1 /home/ubuntu/omron/.venv/bin/python3 miner.py && \
    rm -rf ~/omron/neurons/deployment_layer/*/target/release/build && \
    rm -rf ~/omron/neurons/deployment_layer/*/target/release/deps && \
    rm -rf ~/omron/neurons/deployment_layer/*/target/release/examples && \
    rm -rf ~/omron/neurons/deployment_layer/*/target/release/incremental && \
    rm -rf ~/.bittensor
USER root
RUN cat <<'EOF' > /entrypoint.sh
#!/usr/bin/env bash
set -e
if [ -n "$PUID" ]; then
    if [ "$PUID" = "0" ]; then
        echo "Running as root user"
        /home/ubuntu/omron/.venv/bin/python3 "$@"
    else
        echo "Changing ubuntu user id to $PUID"
        usermod -u "$PUID" ubuntu
        gosu ubuntu /home/ubuntu/omron/.venv/bin/python3 "$@"
    fi
else
    gosu ubuntu /home/ubuntu/omron/.venv/bin/python3 "$@"
fi
EOF
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["-c", "import subprocess; \
    subprocess.run(['/home/ubuntu/omron/.venv/bin/python3', '/home/ubuntu/omron/neurons/miner.py', '--help']); \
    subprocess.run(['/home/ubuntu/omron/.venv/bin/python3', '/home/ubuntu/omron/neurons/validator.py', '--help']);" \
    ]
# Axon server
EXPOSE 8091/tcp
# API server
EXPOSE 8443/tcp
# Prometheus server
EXPOSE 9090/tcp
