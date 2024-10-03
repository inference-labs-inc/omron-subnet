FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/base:jammy

# Install dependencies and some nice to have tools
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    wget \
    curl \
    make \
    clang \
    libssl-dev \
    llvm \
    libudev-dev \
    protobuf-compiler \
    byobu \
    fish \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python dependencies, and some more nice to have tools
RUN pip3 install --upgrade pip
RUN pip3 install bpython ptpython pipdeptree pysnooper coverage
# Copy of requirements.txt (Dockerfile can't copy from parent directory)
RUN pip3 install \
    torch \
    python-dotenv \
    requests \
    numpy==1.26.4 \
    setuptools \
    bittensor==6.9.4 \
    GitPython \
    wandb \
    psutil \
    packaging==24.1 \
    pre-commit \
    attrs==24.2.0

# Install node, npm, and pm2 (version in apt is too old)
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash && \
    export NVM_DIR="$HOME/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" && \
    nvm install 20 && \
    npm install -g pm2

# Install subtensor template
RUN git clone https://github.com/opentensor/bittensor-subnet-template.git && \
    cd bittensor-subnet-template && \
    pip3 install -e .

# Install subtensor
RUN git clone https://github.com/opentensor/subtensor.git && \
    cd subtensor && \
    cargo build --workspace --profile=release --features pow-faucet --manifest-path "Cargo.toml"

# Set flags for local subtensor
ENV WPATH="--wallet.path /root/.bittensor/wallets/"
ENV LOCALNET="--subtensor.chain_endpoint ws://127.0.0.1:9946"

# Install btcli (separate package as of bittensor v8.0.0)
RUN pip3 install bittensor-cli

# Create wallets
RUN btcli wallet new_coldkey --no-use-password --n-words 15 $WPATH --wallet.name owner && \
    btcli wallet new_hotkey  --no-use-password --n-words 15 $WPATH --wallet.name owner --wallet.hotkey default && \
    btcli wallet new_coldkey --no-use-password --n-words 15 $WPATH --wallet.name miner && \
    btcli wallet new_hotkey  --no-use-password --n-words 15 $WPATH --wallet.name miner --wallet.hotkey default && \
    btcli wallet new_coldkey --no-use-password --n-words 15 $WPATH --wallet.name validator && \
    btcli wallet new_hotkey  --no-use-password --n-words 15 $WPATH --wallet.name validator --wallet.hotkey default

# Mint some tokens
RUN cd subtensor && \
    BUILD_BINARY=0 ./scripts/localnet.sh > /dev/null 2>&1 & \
    sleep 5 && \
    yes | btcli wallet faucet --wallet.name owner $WPATH $LOCALNET & \
    sleep 30 && \
    pkill -2 localnet ; \
    pkill -2 subtensor

RUN cd subtensor && \
    BUILD_BINARY=0 ./scripts/localnet.sh --no-purge > /dev/null 2>&1 & \
    sleep 5 && \
    yes | btcli wallet faucet --wallet.name miner $WPATH $LOCALNET & \
    sleep 30 && \
    pkill -2 localnet ; \
    pkill -2 subtensor

RUN cd subtensor && \
    BUILD_BINARY=0 ./scripts/localnet.sh --no-purge > /dev/null 2>&1 & \
    sleep 5 && \
    yes | btcli wallet faucet --wallet.name validator $WPATH $LOCALNET & \
    sleep 30 && \
    pkill -2 localnet ; \
    pkill -2 subtensor

# Register the subnet
RUN cd subtensor && \
    BUILD_BINARY=0 ./scripts/localnet.sh --no-purge > /dev/null 2>&1 & \
    sleep 5 && \
    btcli subnet create   --wallet.name owner     --no-prompt              --wallet.hotkey default $WPATH $LOCALNET && \
    btcli subnet register --wallet.name miner     --no-prompt --netuid 1   --wallet.hotkey default $WPATH $LOCALNET && \
    btcli subnet register --wallet.name validator --no-prompt --netuid 1   --wallet.hotkey default $WPATH $LOCALNET && \
    btcli root nominate   --wallet.name validator --no-prompt              --wallet.hotkey default $WPATH $LOCALNET && \
    btcli stake add       --wallet.name validator --no-prompt --amount 100 --wallet.hotkey default $WPATH $LOCALNET && \
    pkill -2 localnet ; \
    pkill -2 subtensor

# Add scripts to start and stop the localnet, and aliases for common btcli commands
RUN echo "#!/usr/bin/env bash" > /start_localnet.sh && \
    echo "cd /subtensor" >> /start_localnet.sh && \
    echo "BUILD_BINARY=0 ./scripts/localnet.sh --no-purge > /dev/null 2>&1 &" >> /start_localnet.sh && \
    chmod +x /start_localnet.sh && \
    echo "#!/usr/bin/env bash" > /stop_localnet.sh && \
    echo "pkill -2 localnet" >> /stop_localnet.sh && \
    echo "pkill -2 subtensor" >> /stop_localnet.sh && \
    chmod +x /stop_localnet.sh && \
    echo 'alias btcliwo="btcli wallet overview '"$WPATH"' '"$LOCALNET"'"' >> ~/.bashrc && \
    fish -c 'alias --save btcliwo="btcli wallet overview '"$WPATH"' '"$LOCALNET"'"' && \
    echo 'alias btclisl="btcli subnet list '"$LOCALNET"'"' >> ~/.bashrc && \
    fish -c 'alias --save btclisl="btcli subnet list '"$LOCALNET"'"'
