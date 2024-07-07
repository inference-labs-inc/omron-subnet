# TEE (Trusted Execution Environment) Inferences

TEE (Trusted Execution Environment) is a secure area within a processor that ensures the confidentiality and integrity of code and data loaded inside it. In the context of our project, we utilize TEE for secure and verifiable inferences.

## Overview

Our implementation leverages Intel SGX (Software Guard Extensions) technology to create a trusted execution environment. This allows us to run sensitive computations and handle confidential data in an isolated and protected enclave.

## Key Components

1. **Docker Container**: We use a specialized Docker image (`intelanalytics/bigdl-ppml-trusted-bigdl-llm-gramine-ref:2.4.0-SNAPSHOT`) that includes the necessary tools and libraries for TEE operations.

2. **Gramine Library OS**: Gramine is used to run unmodified applications in Intel SGX enclaves, providing a secure runtime environment for our model inferences.

3. **Controller and Worker Services**: The system is divided into controller and worker services, both running within the TEE.

## Setup and Deployment

### Controller Service

The controller service manages the overall workflow and coordinates with worker nodes. It is configured as follows:

- Uses the `intelanalytics/bigdl-ppml-trusted-bigdl-llm-gramine-ref:2.4.0-SNAPSHOT` Docker image
- Exposes ports 21005 and 8000 for communication
- Mounts the SGX device and AESMD socket for SGX operations

### Worker Service

Worker nodes perform the actual model inferences within the TEE. Key configurations include:

- Uses the same Docker image as the controller
- Connects to the controller service
- Mounts the model data and SGX-related devices
- Configures resources such as memory and CPU limits

## Verification Process

The TEE ensures that inferences are performed in a secure and verifiable manner. This includes:

1. Remote attestation to verify the integrity of the TEE
2. Secure loading of the model into the enclave
3. Protected execution of inferences
4. Verifiable output generation

## Integration with Validator

The validator interacts with the TEE to request and verify inferences. This process involves:

1. Sending inference requests to the TEE
2. Receiving results and proofs from the TEE
3. Verifying the integrity and authenticity of the results
4. Updating scores based on the verified inferences

By utilizing TEE, we ensure that our inference process is secure, tamper-proof, and verifiable, which is crucial for maintaining the integrity of our decentralized AI system.
