# Matrix Multiplication Competition

> This is an example competition **this will not be deployed to mainnet**

This competition challenges miners to implement matrix multiplication in a zero-knowledge circuit.

## Problem Description

Implement a circuit that performs matrix multiplication between:

- Matrix 1: 2x5 matrix
- Matrix 2: 5x3 matrix

The result should be a 2x3 matrix.

## Input Format

- Input is a flattened vector of 25 elements
- First 10 elements represent Matrix 1 (2x5)
- Next 15 elements represent Matrix 2 (5x3)

## Output Format

- Output should be a flattened vector of 6 elements
- Represents the 2x3 result matrix

## Required Files

1. vk.key - Verification key
2. pk.key - Proving key
3. settings.json - Model configuration
4. model.compiled - Compiled circuit
5. network.onnx - ONNX model
6. input.json - Input specification

## Scoring

- Accuracy Weight: 80%
- Remaining 20% split between:
  - Proof size optimization
  - Response time optimization

## Example settings.json

See sample_settings.json in this directory.

## Timeline

- Start: February 3, 2025
- End: February 5, 2025

## Tips

1. Use EZKL for circuit compilation
2. Ensure your circuit handles the matrix shapes correctly
3. Test with random inputs before submitting
4. Optimize your circuit for both accuracy and efficiency
