<div align="center">

# **Omron SN2 (𝜏, β)**

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg?logo=discord)](https://discord.gg/bittensor)

### Proof of Inference

[Discord](https://discord.gg/bittensor) • [X](https://twitter.com/omron_ai) • [Inference Labs](https://twitter.com/inference_labs) • [Explorer](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)

</div>

The Omron subnetwork runs verified inferences on staking and restaking optimization models to generate staking and restaking strategy recommendations in a secure and verifiable manner.

## Quickstart

Run the below command to install Omron and it's dependencies.

```console
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/inference-labs-inc/omron-subnet/main/setup.sh)"
```

[See full setup guide →](docs/shared_setup_steps.md)

### Register on the SN

```console
btcli subnet register --subtensor.network finney --netuid 2 --wallet.name {your_coldkey} --wallet.hotkey {your_hotkey}

```

### Run the miner

```console
cd neurons
pm2 start miner.py --name miner --interpreter python3 -- \
--netuid 2 \
--wallet.name {your_miner_key_name} \
--wallet.hotkey {your_miner_hotkey_name}
```

### Run the validator

```console
cd neurons
pm2 start validator.py --name validator --interpreter python3 -- \
--netuid 2 \
--wallet.name {validater_key_name} \
--wallet.hotkey {validator_hot_key_name}
```

## Miner

Miners contribute to this subnet by providing compute to generate output from, and prove AI model inferences. Miners receive workloads from validators in the form of input data, perform verified inferences on those inputs and respond with output along with a zero knowledge proof of inference.

### Hardware requirements

#### Minimum

| Component | Requirement |
| --------- | ---------- |
| CPU | 8 core 3.2GHz |
| RAM | 32GB |
| Network Up | 400Mbps |
| Network Down | 400Mbps |
| Storage | 100GB |

#### Recommended

> [!INFO]
> Exceeding these requirements in terms of storage, network and CPU speed will most likely result in higher rewards due to performance incentivization.

| Component | Recommendation |
| --------- | ---------- |
| CPU | 8 core 3.6GHz |
| RAM | 64GB |
| Network Up | 1Gbps |
| Network Down | 1Gbps |
| Storage | 400GB |
| Storage Medium | SSD |

## Validator

Validators are responsible for verifying model outputs as provided by miners, and updating that miner's score based on the verification results.

### Hardware requirements

#### Minimum

| Component | Requirement |
| --------- | ---------- |
| CPU | 8 core 3.2GHz |
| RAM | 16GB |
| Network Up | 400Mbps |
| Network Down | 400Mbps |
| Storage | 100GB |

#### Recommended

| Component | Recommendation |
| --------- | ---------- |
| CPU | 8 core 3.4GHz |
| RAM | 32GB |
| Network Up | 1Gbps |
| Network Down | 1Gbps |
| Storage | 300GB |
| Storage Medium | SSD |
