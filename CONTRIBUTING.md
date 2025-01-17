# Developer Guide

This document provides a guide for developers who contribute to the Omron subnet.

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

For local development, we suggest running a [local devnet node](https://github.com/inference-labs-inc/bittensor-devnet).

### Miner

```sh
python ./neurons/miner.py --localnet
```

### Validator

```sh
python ./neurons/validator.py --localnet
```
