FROM ghcr.io/inference-labs-inc/bittensor-devcontainer:latest

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
