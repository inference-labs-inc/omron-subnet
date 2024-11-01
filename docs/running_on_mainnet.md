<div align="center">

# Running on Mainnet

</div>

## Setup

Please find relevant setup documentation over in the [`shared_setup_steps.md`] file. These steps will prepare the miner and validator for use in the following steps.

## Mining

Run the following command to start a miner on mainnet

```console
cd neurons
pm2 start miner.py --name miner --interpreter python3 -- \
--netuid 2 \
--wallet.name {your_miner_key_name} \
--wallet.hotkey {your_miner_hotkey_name}
```

[View all acceptable CLI arguments →]

## Validating

Run the following command to start a validator on mainnet

```console
cd neurons
pm2 start validator.py --name validator --interpreter python3 -- \
--netuid 2 \
--wallet.name {your_validator_key_name} \
--wallet.hotkey {your_validator_hotkey_name}
```

[View all acceptable CLI arguments →]

[View all acceptable CLI arguments →]: ./command_line_arguments.md
[`shared_setup_steps.md`]: ./shared_setup_steps.md
