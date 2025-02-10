# Competition Quick Start Guide

## Prerequisites

- Python 3.8+
- Storage provider credentials (R2/S3)
- Registration on subnet 2 (use `btcli register` if not already registered)

> [!NOTE]
> While miners can run on any platform, testing on macOS arm64 is recommended since validators use this architecture for evaluation. This helps ensure your circuit will perform consistently when being evaluated.

## Getting Started

1. **Configure Storage**

The following variables configure remote storage for your circuit files.

```bash
export STORAGE_PROVIDER="r2"  # or "s3"
export STORAGE_BUCKET="your-bucket"
export STORAGE_ACCOUNT_ID="your-account-id"
export STORAGE_ACCESS_KEY="your-access-key"
export STORAGE_SECRET_KEY="your-secret-key"
export STORAGE_REGION="auto"  # or AWS region for S3
```

2. **Start Miner**

Start the miner process normally using `pm2` or your preferred process manager.

## Circuit Submission Flow

1. **Prepare Circuit Files**

   - Compile your circuit
   - Generate proving/verification keys
   - Test locally with sample inputs

2. **Deploy Circuit**

- Move your files into the `competition_circuit/` directory.
- The miner process will automatically detect your circuit
- New circuit files are uploaded to remote storage
- Verification key is committed to the chain

3. **Monitor Evaluation**
   - Watch logs for validator requests for your circuit
   - Track SOTA competition metrics on https://accelerator.omron.ai
