# Developers Guide

This document provides a guide for developers who contributes to the project.

## Adding Dependencies

We use `pip-tools` to manage dependencies. To add new dependencies, follow the steps below:

1. Add the package to `requirements.in` file.
2. Compile dependencies:

```sh
pip-compile --generate-hashes requirements.in
```

3. Sync dependencies:

```sh
pip-sync
```

## Updating Dependencies

To force pip-compile to update all packages in an existing `requirements.txt`, run `pip-compile --upgrade`.

```sh
# only update the bittensor package
$ pip-compile --upgrade-package bittensor

# update both the bittensor and requests packages
$ pip-compile --upgrade-package bittensor --upgrade-package requests

# update the bittensor package to the latest, and requests to v2.0.0
$ pip-compile --upgrade-package bittensor --upgrade-package requests==2.0.0
```

## Running Locally for Development

For local development, we suggest you are running a [local devnet node](https://github.com/inference-labs-inc/bittensor-devnet).

Running miner locally for development:

```sh
python miner.py \
    --netuid 1 \
    --subtensor.chain_endpoint ws://127.0.0.1:9946 \
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
    --subtensor.chain_endpoint ws://127.0.0.1:9946 \
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
