FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/base:noble

# Install dependencies and some nice to have tools
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
    byobu \
    fish \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install subtensor
RUN git clone https://github.com/opentensor/subtensor.git && \
    cd subtensor && \
    git checkout v1.1.6 && \
    cargo build --workspace --profile=release --features pow-faucet --manifest-path "Cargo.toml"

# Install Jolt
ENV RUST_TOOLCHAIN=nightly-2024-09-30
RUN rustup toolchain install ${RUST_TOOLCHAIN} && \
    cargo +${RUST_TOOLCHAIN} install --git https://github.com/a16z/jolt --force --bins jolt

# Install node et al.
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash && \
    export NVM_DIR="/root/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" && \
    nvm install 20 && \
    npm install --prefix /root/.snarkjs snarkjs@0.7.4

# Use a venv because of https://peps.python.org/pep-0668/
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/python3 -m pip install --upgrade pip
ENV PATH="/opt/venv/bin:${PATH}"

# Install Python dependencies, and some more nice to have tools
COPY requirements.txt /opt/omron/requirements.txt
RUN TORCH_VERSION=$(grep "^torch" /opt/omron/requirements.txt) && \
    pip3 install $TORCH_VERSION --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install -r /opt/omron/requirements.txt
RUN pip3 install bittensor-cli==8.3.1 bpython ptpython pipdeptree pysnooper coverage

# Set flags for local subtensor
ENV WPATH="--wallet.path /root/.bittensor/wallets/"
ENV LOCALNET="--subtensor.chain_endpoint ws://127.0.0.1:9946"

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
    sleep 60 && \
    pkill -2 localnet ; \
    pkill -2 subtensor ; \
    sleep 1

RUN cd subtensor && \
    BUILD_BINARY=0 ./scripts/localnet.sh --no-purge > /dev/null 2>&1 & \
    sleep 5 && \
    yes | btcli wallet faucet --wallet.name miner $WPATH $LOCALNET & \
    sleep 60 && \
    pkill -2 localnet ; \
    pkill -2 subtensor ; \
    sleep 1

RUN cd subtensor && \
    BUILD_BINARY=0 ./scripts/localnet.sh --no-purge > /dev/null 2>&1 & \
    sleep 5 && \
    yes | btcli wallet faucet --wallet.name validator $WPATH $LOCALNET & \
    sleep 60 && \
    pkill -2 localnet ; \
    pkill -2 subtensor ; \
    sleep 1

# Register the subnet
RUN cd subtensor && \
    BUILD_BINARY=0 ./scripts/localnet.sh --no-purge > /dev/null 2>&1 & \
    sleep 5 && \
    btcli subnet create   --wallet.name owner     --no-prompt                                      $WPATH $LOCALNET && \
    btcli subnet register --wallet.name miner     --no-prompt --netuid 1   --wallet.hotkey default $WPATH $LOCALNET && \
    btcli subnet register --wallet.name validator --no-prompt --netuid 1   --wallet.hotkey default $WPATH $LOCALNET && \
    btcli root nominate   --wallet.name validator --no-prompt              --wallet.hotkey default $WPATH $LOCALNET && \
    btcli stake add       --wallet.name validator --no-prompt --amount 100 --wallet.hotkey default $WPATH $LOCALNET && \
    pkill -2 localnet ; \
    pkill -2 subtensor ; \
    sleep 1

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
    fish -c 'alias --save btclisl="btcli subnet list '"$LOCALNET"'"' && \
    echo "source /opt/venv/bin/activate" >> ~/.bashrc && \
    echo "source /opt/venv/bin/activate.fish" >> ~/.config/fish/config.fish
