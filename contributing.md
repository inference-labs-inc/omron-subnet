# Developers Guide

This document provides a guide for developers who contributes to the project.

## Adding Dependencies

We use `uv` to manage dependencies. To add new dependencies, follow the steps below:

1. Add the package to `pyproject.toml`:

```sh
uv add <package-name>
```

2. Lock dependencies and generate `requirements.txt`:

```sh
uv lock
uv export -o requirements.txt
```

3. Sync dependencies:

```sh
uv sync --locked
```

## Updating Dependencies

To force uv to update all packages in an existing `pyproject.toml`, run `uv sync --upgrade`.

```sh
# only update the bittensor package
$ uv sync --upgrade-package bittensor

# update both the bittensor and requests packages
$ uv sync --upgrade-package bittensor --upgrade-package requests

# update the bittensor package to the latest, and requests to v2.0.0
$ uv sync --upgrade-package bittensor --upgrade-package requests==2.0.0
```

## Running Locally for Development

For local development, we suggest you are running a [local devnet node](https://github.com/inference-labs-inc/bittensor-devnet).

Running miner locally for development:

```sh
python miner.py \
    --netuid 1 \
    --subtensor.chain_endpoint ws://127.0.0.1:9944 \
    --wallet.name miner \
    --wallet.hotkey default \
    --subtensor.network local \
    --eth_wallet 0x002 \
    --axon.ip 127.0.0.1 \
    --axon.external_ip 127.0.0.1 \
    --disable-wandb true \
    --verbose true \
    --disable_blacklist true \
    --no-auto-update true
```

Running validator locally for development:

```sh
python validator.py \
    --netuid 1 \
    --subtensor.chain_endpoint ws://127.0.0.1:9944 \
    --wallet.name validator \
    --wallet.hotkey default \
    --subtensor.network local \
    --eth_wallet 0x001 \
    --timeout 240 \
    --disable-wandb true \
    --verbose true \
    --disable_blacklist true \
    --no-auto-update true
```
