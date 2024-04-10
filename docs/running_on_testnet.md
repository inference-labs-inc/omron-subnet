<div align="center">

# Running on Testnet

</div>

## Setup

Please find relevant setup documentation over in the [`shared_setup_steps.md`] file. These steps will prepare the miner and validator for use in the following steps.

## Mining

Run the following command to start a miner on testnet

```console
cd neurons
pm2 start miner.py --name miner --interpreter python3 -- \
--netuid 118 \
--wallet.name {your_miner_key_name} \
--wallet.hotkey {your_miner_hotkey_name} \
--subtensor.network test
```

## Validating

Run the following command to start a validator on testnet

```console
cd neurons
pm2 start validator.py --name validator --interpreter python3 -- \
--netuid 118 \
--wallet.name {your_validator_key_name} \
--wallet.hotkey {your_validator_hotkey_name} \
--subtensor.network test
```

[`shared_setup_steps.md`]: ./shared_setup_steps.md
