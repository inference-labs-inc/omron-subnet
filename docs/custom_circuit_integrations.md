# Custom Circuit Integrations

The purpose of this document is to provide an overview of how third parties can integrate their own zero-knowledge circuits into the Omron subnet.

## Adding a new circuit

To add a new circuit into Omron, the following high-level steps are required:

1. Open a PR on [this repository] describing the new circuit.
2. The PR will be reviewed by core contributors, and merged into the main branch upon acceptance.
3. A release will be created by the Omron team and the new circuit will be available to the network in approximately 24 hours.

Please see the below sections for a comprehensive tutorial on how to add a new circuit.

### Build your circuit

To build your circuits, you will need to use one of the supported proof systems or propose integration with a new proof system.

Building your circuit is easiest through the EZKL proof system. Please see the [EZKL documentation](https://github.com/zkonduit/ezkl) for detailed instructions around setup and usage.

> [!IMPORTANT]
>
> One of the inputs to your circuit must be an F32 field named `nonce`. Though it does not need to be used in the circuit, it is required to be present in the input to allow the subnet to secure it's internal state. If you are unsure about this step, please reach out to the Omron team for assistance.

Once you have built your circuit, you will possess the following files which are necessary for circuit integration:

- `vk.key`
- `pk.key`
- `model.compiled`
- `settings.json`

> [!NOTE]
>
> Please reach out using the [Seeking Assistance](#seeking-assistance) section below to discuss your proposal if you are interested in adding a new proof system.

### Gather the necessary files

Omron requires that circuits are added with several supporting files to facilitate the circuit's integration into the subnet.

| File            | Description                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------- |
| `input.py`      | The input file for the circuit. This file is responsible for generating the input for the circuit.   |
| `metadata.json` | The metadata file for the circuit. This file is responsible for describing the circuit's properties. |
| `settings.json` | The settings file for the circuit. Modifications are required after this file is generated.          |

#### `input.py`

This file is responsible for handling inputs to the circuit, both from real world requests and from benchmarking requests. `input.py` must inherit from [`BaseInput`] and an example of how to implement this can be found in the [input.py file for one of the existing circuits].

Ensure that the class created is registered to the input registry, by including this decorator above the class definition:

```python
@InputRegistry.register(
    "<hash_of_circuit_verification_key>"
)
class CircuitInput(BaseInput):
    ...
```

There are three primary functions which must be implemented.

| Function   | Description                                                                                        |
| ---------- | -------------------------------------------------------------------------------------------------- |
| `generate` | Generates **randomized**, reference inputs for the circuit. These are used for benchmark requests. |
| `validate` | Validates the external inputs for the circuit.                                                     |
| `process`  | Processes external input for the circuit.                                                          |

> [!CAUTION]
>
> `input.py` **must** implement a unique key field named `nonce`. This field is required for security reasons and is used internally by the subnet.

#### `metadata.json`

This file is responsible for describing the circuit's properties, and contains both optional and required fields.

| Field             | Type            | Description                                                                          | Required |
| ----------------- | --------------- | ------------------------------------------------------------------------------------ | -------- |
| `name`            | `string`        | The name of the circuit.                                                             | ✅       |
| `description`     | `string`        | A short description of the circuit.                                                  | ✅       |
| `author`          | `string`        | The author of the circuit.                                                           | ✅       |
| `version`         | `string`        | The version of the circuit.                                                          | ✅       |
| `proof_system`    | [`ProofSystem`] | The proof system used by the circuit.                                                | ✅       |
| `type`            | [`CircuitType`] | The type of circuit.                                                                 | ✅       |
| `external_files`  | `dict`          | A dictionary of external files required by the circuit.                              | ✅       |
| `netuid`          | `int`           | For Proof of Weights, the netuid of the target subnet.                               | ❌       |
| `weights_version` | `int`           | For Proof of Weights, the version of subnet weights that the circuit corresponds to. | ❌       |

[See it's class definition here for more information](https://github.com/inference-labs-inc/omron-subnet/blob/main/neurons/execution_layer/circuit.py#L100)

#### `settings.json`

This file is generated automatically by the EZKL proof system (if using EZKL). An additional step is required to modify the settings file after it is generated.

Locate the `model_input_scales` field in the `settings.json` file. This field is a list of floats, where each float corresponds to the scale of the input for a specific input field.

Add a new field below this list called `model_input_types`. This should look the exact same as the `model_input_scales`, but instead: This field is a list of strings, where each string corresponds to the type of the input for a specific input field. A list of valid types can be found [here](https://github.com/zkonduit/ezkl/blob/f35688917d09806196fdedc7fc69804357363183/src/circuit/ops/mod.rs#L86). If you are unsure about this step, please reach out to the Omron team for assistance.

### Create the circuit directory

Use the following command to generate the `SHA256` hash of the circuit's verification key, which will be used as the circuit's folder name and unique identifier.

```sh
sha256sum <path_to_circuit_verification_key>
```

Copy this hash, and create a new directory within the `neurons/deployment_layer/` folder named `model_<hash>`.

Ensure all the following files are copied into this directory or are listed in the `external_files` field of the `metadata.json` file:

- `input.py`
- `metadata.json`
- `settings.json`
- `vk.key`
- `pk.key`
- `model.compiled`

### Adding benchmark requests

The last step after adding your circuit is to add an allocation of benchmark requests to the circuit. This is done through a modification to the `neurons/constants.py` file. In the `CIRCUIT_WEIGHTS` dictionary, add a new key-value pair where the key is the `SHA256` hash of the circuit's verification key, and the value is the percentage of requests that should be allocated to the circuit. Percentages should be between 0 and 1 and do not need to sum to 1. A reasonable starting point is 0.20 for each circuit. The omron team will adjust these weights as necessary to ensure a healthy distribution of requests across all circuits.

A higher percentage will result in more requests being allocated to the circuit, increasing the circuit's optimization pressure.

### Updating Existing Circuits

To update an existing circuit:

1. Follow the same steps as adding a new circuit, which will result in a new verification key hash
2. In the `CIRCUIT_WEIGHTS` dictionary, set the old circuit's allocation to 0.0
3. Add the new circuit with the desired benchmark allocation
4. Open a PR with these changes

## Querying a circuit

To query a circuit, the following high-level steps are required:

1. Fetch a validator's axon IP and port from the blockchain.
2. Fetch the validator's latest certificate hash from the blockchain.
3. Connect to the validator's WebSocket server using the provided IP and port.
4. Upon connection, compare the DER certificate hash to the validator's latest certificate hash. **If they do not match, disconnect** and try again.
5. Once connected, send an RPC request to the validator to commence circuit execution.
6. Wait for the validator to respond with the circuit's output or provide an error message.

### Selecting a validator

Bittensor subnets work based on a validator<->miner relationship. Validators are responsible for evaluating the miner's responses to it's requests. As such, to query into the Omron subnet, you will need to connect to a validator directly.

For those unfamiliar with Bittensor validators, please reach out using the [Seeking Assistance](#seeking-assistance) section below. We'll be happy to work with you to get connected through a subnet validator.

For those who have a validator to connect with, please proceed to the next step.

### Connecting

#### Through Omron's API

For those who are interested in querying a circuit through Omron's API, please reach out using the [Seeking Assistance](#seeking-assistance) section below. We'll be happy to work with you to get connected.

#### Through a validator

> [!NOTE]
>
> This section is subject to change as we continue to expand the validator API.

Validator APIs are available at `wss://<validator_ip>:<validator_port>/rpc`. These are WebSocket servers which use the [JSON-RPC 2.0](https://www.jsonrpc.org/specification) protocol.

Omron implements the `wss` protocol for all validator APIs, which use self-signed certificates for connection security. Each validator commits their self-signed certificate to the blockchain as a `SHA256` hash. To perform verification on the connection please see the below example, which demonstrates connection validation through the `wss` protocol.

[`verify_ssl.py`]

> [!IMPORTANT]
>
> You **must** use the `wss` protocol when connecting to a validator, and confirm that the DER certificate hash of the validator matches the validator's latest certificate hash to confirm the connection is secure.

Additionally, several header fields are required to be included in the connection request to authenticate the request. These fields are:

| Key             | Required | Description                                                                   |
| --------------- | -------- | ----------------------------------------------------------------------------- |
| `x-origin-ss58` | Yes      | The SS58 address of the Origin.                                               |
| `x-signature`   | Yes      | A SS58 signature made by `x-origin-ss58`, containing the `x-timestamp` value. |
| `x-timestamp`   | Yes      | A Unix timestamp of when the request was created.                             |

Upon connection, the validator will check it's whitelist of valid origins, and if the provided `x-origin-ss58` is not in the whitelist or the provided signature is not valid, the connection will be rejected.

### Requesting a circuit execution

To request a circuit execution, send an RPC request to the validator's WebSocket server. This request should be a JSON object with the following fields:

| Key       | Required | Description                                          |
| --------- | -------- | ---------------------------------------------------- |
| `input`   | Yes      | Input to the circuit, as a JSON object.              |
| `circuit` | Yes      | The unique hash of the circuit to use for the proof. |

```json
{
  "jsonrpc": "2.0",
  "method": "omron.proof_of_computation",
  "params": {
    "input": {...},
    "circuit": "..."
  },
  "id": 1
}
```

### Receiving a circuit execution

Once complete, the validator will respond with the circuit's output or provide an error message.

#### Success

```json
{
  "jsonrpc": "2.0",
  "result": {
    "output": {...},
    "proof": "..."
  },
  "id": 1
}
```

#### Error

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": ...,
    "message": "...",
    "data": {...}
    },
  "id": 1
}
```

## Seeking Assistance

If you have any questions or need assistance with the process of adding a new circuit, please reach out to the Omron team through the subnet 2 channel within the Bittensor Discord using the badge below, or via our contact form on [omron.ai].

[![Discord](https://img.shields.io/badge/Join-gray?style=for-the-badge&logo=Discord&logoColor=blue&link=https%3A%2F%2Fdiscord.gg%2FBECadXnAtE)](https://discord.gg/BECadXnAtE)

[this repository]: https://github.com/inference-labs-inc/omron-subnet
[omron.ai]: https://omron.ai
[`ProofSystem`]: https://github.com/inference-labs-inc/omron-subnet/blob/main/neurons/execution_layer/circuit.py#L22
[`CircuitType`]: https://github.com/inference-labs-inc/omron-subnet/blob/main/neurons/execution_layer/circuit.py#L13
[`BaseInput`]: https://github.com/inference-labs-inc/omron-subnet/blob/main/neurons/execution_layer/base_input.py#L6
[input.py file for one of the existing circuits]: https://github.com/inference-labs-inc/omron-subnet/blob/b321b84519456fbe7f17adbda629c1c92bda32bd/neurons/deployment_layer/model_33b92394b18412622adad75733a6fc659b4e202b01ee8a5465958a6bad8ded62/input.py
[`verify_ssl.py`]: https://github.com/inference-labs-inc/omron-subnet/blob/scripts/neurons/scripts/verify_ssl.py
