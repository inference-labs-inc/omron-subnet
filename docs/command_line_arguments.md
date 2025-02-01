# Command Line Arguments

These are options configurable via command line arguments, when running miner or validator software.

## Custom Arguments

Arguments that are present within the Omron miner and validator software. The below arguments apply to both miner and validator software.

| Argument           | Required | Default | Accepted Values | Description                                                                |
| ------------------ | :------: | ------- | --------------- | -------------------------------------------------------------------------- |
| `--netuid`         |   Yes    | `2`     | Integer         | The subnet UID                                                             |
| `--no-auto-update` |    No    | `False` | `True`, `False` | Whether automatic update should be disabled.                               |
| `--wandb-key`      |    No    | `None`  | String          | A WandB API key for logging purposes                                       |
| `--disable-wandb`  |    No    | `False` | `True`, `False` | Whether to disable WandB logging.                                          |
| `--dev`            |    No    | `False` | `True`, `False` | Whether to run the software in development mode. **For internal use only** |
| `--localnet`       |    No    | `False` | `True`, `False` | Whether to run the validator in localnet mode.                             |

### Miner specific arguments

The below arguments are specific to miner software and have no effect on validator software.

| Argument              | Required | Default | Accepted Values | Description                                                  |
| --------------------- | :------: | ------- | --------------- | ------------------------------------------------------------ |
| `--disable-blacklist` |    No    | `False` | `True`, `False` | Disables request filtering and allows all incoming requests.  |

### Validator specific arguments

The below arguments are specific to validator software and have no effect on miner software.

| Argument                              | Required | Default   | Accepted Values | Description                                                                                                                                                                                 |
| ------------------------------------- | :------: | --------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--blocks-per-epoch`                  |    No    | `50`      | Integer         | The number of blocks validators wait to set weights on-chain                                                                                                                                |
| `--enable-pow`                        |    No    | `False`   | `True`, `False` | Whether on-chain proof of weights is enabled                                                                                                                                                |
| `--pow-target-interval`               |    No    | `1000`    | Integer         | The target block interval for committing proof of weights to the chain                                                                                                                      |
| `--ignore-external-requests`          |    No    | `True`    | `True`, `False` | Whether the validator should ignore external requests through it's API.                                                                                                                     |
| `--external-api-port`                 |    No    | `8443`    | Integer         | The port for the validator's external API.                                                                                                                                                  |
| `--external-api-workers`              |    No    | `1`       | Integer         | The number of workers for the validator's external API.                                                                                                                                     |
| `--external-api-host`                 |    No    | `0.0.0.0` | String          | The host for the validator's external API.                                                                                                                                                  |
| `--do-not-verify-external-signatures` |    No    | `False`   | `True`, `False` | External PoW requests are signed by validator's (sender's) wallet. By default, these are checked to ensure legitimacy. This should only be disabled in controlled development environments. |
| `--prometheus-monitoring`             |    No    | `False`   | `True`, `False` | Whether to enable sering of metrics for Prometheus monitoring.                                                                                                                              |
| `--prometheus-port`                   |    No    | `9090`    | Integer         | The port for the Prometheus data source.                                                                                                                                                    |
| `--serve-axon`                        |    No    | `False`   | `True`, `False` | Whether to serve the axon displaying your API information.                                                                                                                                  |

## Built-in Arguments

Arguments that are built into bittensor packages, and can be provided to change the behavior of bittensor related functionalities.

### Wallet

Bittensor wallet configuration options.

[View in code →](https://github.com/opentensor/bittensor/blob/master/bittensor/wallet.py#L134)

| Argument          | Required | Default                 | Accepted Values | Description                                                                                               |
| ----------------- | :------: | ----------------------- | --------------- | --------------------------------------------------------------------------------------------------------- |
| `--no_prompt`     |    No    | `False`                 | `True`, `False` | Set true to avoid prompting the user.                                                                     |
| `--wallet.name`   |    No    | `default`               | String          | The name of the wallet to unlock for running bittensor (name "mock" is reserved for mocking this wallet). |
| `--wallet.hotkey` |    No    | `default`               | String          | The name of the wallet's hotkey.                                                                          |
| `--wallet.path`   |    No    | `~/.bittensor/wallets/` | String          | The path to your bittensor wallets.                                                                       |

### Subtensor

Bittensor subtensor configuration options.

[View in code →](https://github.com/opentensor/bittensor/blob/master/bittensor/subtensor.py#L170)

| Argument                     | Required | Default            | Accepted Values                      | Description                                                                                                                |
| ---------------------------- | :------: | ------------------ | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `--subtensor.network`        |    No    | `finney`           | `finney`, `test`, `archive`, `local`   | The subtensor network to connect to. Overrides `--subtensor.chain_endpoint` with a default node from the selected network. |
| `--subtensor.chain_endpoint` |    No    | Depends on network | String                               | The specific blockchain endpoint to connect to. Overrides the network default endpoint if set.                              |
| `--subtensor._mock`          |    No    | `False`            | `True`, `False`                      | If true, uses a mocked connection to the chain for testing purposes.                                                       |

### Axon

Bittensor Axon configuration options.

[View in code →](https://github.com/opentensor/bittensor/blob/master/bittensor/axon.py#L600)

| Argument               | Required | Default | Accepted Values | Description                                                                               |
| ---------------------- | :------: | ------- | --------------- | ----------------------------------------------------------------------------------------- |
| `--axon.port`          |    No    | 8091    | Integer         | The local port this axon endpoint is bound to.                                            |
| `--axon.ip`            |    No    | `[::]`  | String          | The local IP this axon binds to.                                                          |
| `--axon.external_port` |    No    | None    | Integer         | The public port this axon broadcasts to the network.                                      |
| `--axon.external_ip`   |    No    | None    | String          | The external IP this axon broadcasts to the network.                                      |

### Logging

Bittensor logging configuration options.

[View in code →](https://github.com/opentensor/bittensor/blob/master/bittensor/btlogging/loggingmachine.py#L334)

| Argument                | Required | Default              | Accepted Values | Description                                |
| ----------------------- | :------: | -------------------- | --------------- | ------------------------------------------ |
| `--logging.debug`       |    No    | `False`              | `True`, `False` | Turn on bittensor debugging information.   |
| `--logging.trace`       |    No    | `False`              | `True`, `False` | Turn on bittensor trace level information. |
| `--logging.record_log`  |    No    | `False`              | `True`, `False` | Turns on logging to file.                   |
| `--logging.logging_dir` |    No    | `~/.bittensor/logs/` | String          | Logging default root directory.            |
