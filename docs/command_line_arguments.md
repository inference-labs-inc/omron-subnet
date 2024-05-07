# Command Line Arguments

These are options configurable via command line arguments, when running miner or validator software.

## Custom Arguments

Arguments that are present within the Omron miner and validator software.

| Argument | Required | Default | Accepted Values | Description |
| --- | :-: | --- | --- | --- |
| `--netuid` | Yes | `2` | Integer | The subnet UID |
| `--auto-update` | No | `True` | `True`, `False` | Whether to automatically check for and perform software updates. |
| `--blocks-per-epoch` | No | `50` | Integer | The number of blocks validators wait to set weights on-chain |
| `--wandb-key` | No | `None` | String | A WandB API key for logging purposes |
| `--disable-wandb` | No | `False` | `True`, `False` | Whether to disable WandB logging. |
| `--dev` | No | `False` | `True`, `False` | Whether to run the software in development mode. **For internal use only** |

## Built-in Arguments

Arguments that are built into bittensor packages, and can be provided to change the behavior of bittensor related functionalities.

### Wallet

Bittensor wallet configuration options.

[View in code →](https://github.com/opentensor/bittensor/blob/master/bittensor/wallet.py#L134)

| Argument | Required | Default | Accepted Values | Description |
| --- | :-: | --- | --- | --- |
| `--no_prompt` | No | `False` | `True`, `False` | Set true to avoid prompting the user. |
| `--wallet.name` | No | `default` | String | The name of the wallet to unlock for running bittensor (name "mock" is reserved for mocking this wallet). |
| `--wallet.hotkey` | No | `default` | String | The name of the wallet's hotkey. |
| `--wallet.path` | No | `~/.bittensor/wallets/` | String | The path to your bittensor wallets. |

### Subtensor

Bittensor subtensor configuration options.

[View in code →](https://github.com/opentensor/bittensor/blob/master/bittensor/subtensor.py#L170)

| Argument | Required | Default | Accepted Values | Description |
| --- | :-: | --- | --- | --- |
| `--subtensor.network` | No | `finney` | `finney`, `test`, `archive`, `local` | The subtensor network to connect to. Overrides `--subtensor.chain_endpoint` with a default node from the selected network. |
| `--subtensor.chain_endpoint` | No | Depends on network | String | The specific blockchain endpoint to connect to. Overrides the network default endpoint if set. |
| `--subtensor._mock` | No | `False` | `True`, `False` | If true, uses a mocked connection to the chain for testing purposes. |

### Axon

Bittensor Axon configuration options.

[View in code →](https://github.com/opentensor/bittensor/blob/master/bittensor/axon.py#L600)

| Argument | Required | Default | Accepted Values | Description |
| --- | :-: | --- | --- | --- |
| `--axon.port` | No | 8091 | Integer | The local port this axon endpoint is bound to. |
| `--axon.ip` | No | `[::]` | String | The local IP this axon binds to. |
| `--axon.external_port` | No | None | Integer | The public port this axon broadcasts to the network. |
| `--axon.external_ip` | No | None | String | The external IP this axon broadcasts to the network. |
| `--axon.max_workers` | No | 10 | Integer | The maximum number of connection handler threads working simultaneously on this endpoint. |

### Logging

Bittensor logging configuration options.

[View in code →](https://github.com/opentensor/bittensor/blob/master/bittensor/btlogging/loggingmachine.py#L334)

| Argument | Required | Default | Accepted Values | Description |
| --- | :-: | --- | --- | --- |
| `--logging.debug` | No | `False` | `True`, `False` | Turn on bittensor debugging information. |
| `--logging.trace` | No | `False` | `True`, `False` | Turn on bittensor trace level information. |
| `--logging.record_log` | No | `False` | `True`, `False` | Turns on logging to file. |
| `--logging.logging_dir` | No | `~/.bittensor/logs/` | String | Logging default root directory. |
