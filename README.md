# omron-tee

## Table of Contents

1. [Introduction](#introduction)
2. [Demo](#demo)
3. [Features](#features)
4. [Tech Stack](#tech-stack)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)
9. [Team](#team)
10. [Acknowledgements](#acknowledgements)

## Introduction

We have extended the functionality of Omron, Bittensor SN2, to include the use of Trusted Execution Environments (TEEs) as a means of verification and Proof-of-Inference. Miners compete against each other to be the best in performance, and only the top miners receive rewards from the Bittensor network.

Trusted Execution Environments (TEEs) provide a secure way of executing Trusted Apps (TAs). This enables 3rd parties to query models all while the inputs/prompts remain fully private to all parties.

[Omron](https://omron.ai) is a Proof-of-Inference compute cluster deployed on Bittensor. [Omron SN 2](https://x.taostats.io/subnet/2) is the worlds largest zkml proving cluster.

Omron currently supports verified inference through the use of zero knowledge proofs. Models are converted to zk circuits which produce cryptographic proof of execution.

This project was completed as a submission in the Brussels ** dAGI House ** hackathon July 5-7 2024. [DoraHacks Link](https://dorahacks.io/hackathon/dagihouse/buidl)

## Demo

*To be added*

## Features

Validators within the subnet query miners and the following exchange occurs.

1. Validator receives a request for Inference
2. Validator selects a miner, generates a unique nonce.
3. Miner configures secure enclave, requesting Remote Attestation (RA) evidence with the nonce. Miner returns the RA to the validator
4. Validator confirms the RA evidence is valid.
5. Validator established a secure and private communication channel with the miner who then receives data to execute the Inference.
6. Inference is completed and verified by the validator
7. Validators update the [scoring](#scoring) of miners based on their performance

## Scoring

Validators score miners based on a correct and valid response. Additionally, miners are ranked on their response time and performance.

## Tech Stack

The following is a non extensive list

- `bittensor==7.2.0`
- `intel_sgx_2.5.101.3_pv`
- `bigdl-llm:2.5.0-SNAPSHOT`

## Installation

Miners require an Intel CPU compatible with SGX.

**Minimum Hardware Requirements**
- **CPU:** 3.8GHz Intel Xeon-CoffeeLake (E-2174G-Quadcore)
- **RAM:** 32GB DDR4
- **Secure Enclave:** Intel SGX or TDX

For the hackathon we provisioned a custom bare metal machine on [IBM Cloud.](https://cloud.ibm.com/docs/bare-metal?topic=bare-metal-ordering-baremetal-server)

Validators do not require an Intel SGX enabled CPU to verify the computations performed by miners.

## Contributing

Open a PR and we will review. Thanks!

## License

[Apache License Version 2.0](/LICENSE)

## Team

Colin G [@gagichce](https://github.com/gagichce)

HudsonGraeme [@HudsonGraeme](https://github.com/HudsonGraeme)

## Acknowledgements

We acknowledge the support from many sponsors who made the Brussels dAGI hackathon possible. This includes [cyber.Fund](https://cyber.fund/) amoung many others listed [here.](https://dorahacks.io/hackathon/dagihouse/buidl)
