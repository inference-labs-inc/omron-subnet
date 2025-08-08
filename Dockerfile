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
    ffmpeg \
    gosu \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Make directories under opt and set owner to ubuntu
RUN mkdir -p /opt/.cargo /opt/.nvm /opt/.npm /opt/.snarkjs /opt/omron/neurons && \
    chown -R ubuntu:ubuntu /opt && \
    chmod -R 775 /opt/omron && \
    chown root:root /opt

# Use ubuntu user
USER ubuntu
WORKDIR /opt

# Install node et al.
ENV NVM_DIR=/opt/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash && \
    export NVM_DIR="$NVM_DIR" && \
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
    nvm install 20 && \
    nvm use 20 && \
    npm install --prefix /opt/.snarkjs snarkjs@0.7.4 && \
    mkdir -p ~/.local/bin && \
    ln -s "$NVM_DIR/versions/node/$(nvm version)/bin/node" /home/ubuntu/.local/bin/node && \
    ln -s "$NVM_DIR/versions/node/$(nvm version)/bin/npm" /home/ubuntu/.local/bin/npm && \
    chmod -R 775 /opt/.nvm /opt/.npm /opt/.snarkjs
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# Copy omron and install Python dependencies (make sure owner is ubuntu)
COPY --chown=ubuntu:ubuntu --chmod=775 neurons /opt/omron/neurons
COPY --chown=ubuntu:ubuntu --chmod=775 pyproject.toml /opt/omron/pyproject.toml
COPY --chown=ubuntu:ubuntu --chmod=775 uv.lock /opt/omron/uv.lock
RUN pipx install uv && \
    cd /opt/omron && \
    ~/.local/bin/uv sync --frozen --no-dev --compile-bytecode && \
    ~/.local/bin/uv cache clean && \
    echo "source /opt/omron/.venv/bin/activate" >> ~/.bashrc && \
    chmod -R 775 /opt/omron/.venv
ENV PATH="/opt/omron/.venv/bin:${PATH}"

# Set workdir for running miner.py or validator.py and compile circuits
WORKDIR /opt/omron/neurons
ENV OMRON_NO_AUTO_UPDATE=1
RUN OMRON_DOCKER_BUILD=1 /opt/omron/.venv/bin/python3 miner.py && \
    rm -rf ~/.bittensor && \
    rm -rf /tmp/omron
USER root
RUN cat <<'EOF' > /entrypoint.sh
#!/usr/bin/env bash
set -e
if [ -n "$PUID" ]; then
    if [ "$PUID" = "0" ]; then
        echo "Running as root user"
        /opt/omron/.venv/bin/python3 "$@"
    else
        echo "Changing ubuntu user id to $PUID"
        usermod -u "$PUID" ubuntu
        gosu ubuntu /opt/omron/.venv/bin/python3 "$@"
    fi
else
    gosu ubuntu /opt/omron/.venv/bin/python3 "$@"
fi
EOF
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["-c", "import subprocess; \
    subprocess.run(['/opt/omron/.venv/bin/python3', '/opt/omron/neurons/miner.py', '--help']); \
    subprocess.run(['/opt/omron/.venv/bin/python3', '/opt/omron/neurons/validator.py', '--help']);" \
    ]
# Axon server
EXPOSE 8091/tcp
# API server
EXPOSE 8443/tcp
# Prometheus server
EXPOSE 9090/tcp
