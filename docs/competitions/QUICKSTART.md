# Competition Quick Start Guide

## Prerequisites

- Python 3.8+
- Storage provider credentials (R2/S3)
- Registration on subnet 2 (use `btcli register` if not already registered)
- At least 16GB RAM (32GB recommended)
- 50GB free SSD storage
- Apple Silicon (M1/M2/M3) CPU

> [!NOTE]
> While miners can run on any platform, macOS arm64 with Metal acceleration is the recommended configuration since validators use this architecture for evaluation. This helps ensure your circuit will perform consistently when being evaluated.

> [!IMPORTANT]
> Ensure your system meets the minimum resource requirements before proceeding. Circuit compilation and proving can be resource-intensive, especially during initial setup.

## Getting Started

### Configure Storage

Create a `.env` file in your project root with the following variables:

```bash
# Required - choose either R2 or S3
STORAGE_PROVIDER="r2"  # or "s3"
STORAGE_BUCKET="your-bucket"

# For Cloudflare R2
STORAGE_ACCOUNT_ID="your-account-id"
STORAGE_ACCESS_KEY="your-access-key"
STORAGE_SECRET_KEY="your-secret-key"
STORAGE_REGION="auto"

# For AWS S3
# STORAGE_REGION="your-aws-region"
```

### Start the Miner

```bash
pm2 start neurons/miner.py --name omron_miner -- --netuid 2 --wallet.name your_wallet --logging.debug
```

## Circuit Submission Flow

### Prepare Circuit Files

Required files in your circuit directory:

- `vk.key` - Verification key
- `pk.key` - Proving key - **must be less than 50GB**
- `settings.json` - Circuit configuration with required settings:
  ```json
  {
    "run_args": {
      "input_visibility": "Private",
      "output_visibility": "Public",
      "param_visibility": "Private",
      "commitment": "KZG"
    }
  }
  ```
- `model.compiled` - Compiled model file

For an example of how to compile this circuit, see the following.

> [!NOTE]
> This script generates it's own model and is for demonstration purposes only. To compile your own model based on the competition template ONNX, please find it within `neurons/_validator/competitions/1/age.onnx` along with an example `input.json` file.

```bash
./neurons/scripts/create_competition_circuit.py
```

### Deploy Circuit

Place circuit files in the `./competition_circuit/` directory:

```bash
mkdir -p competition_circuit/
cp -r your_circuit/* competition_circuit/
```

The miner will automatically:

- Monitor circuit directory for changes
- Upload modified files to R2/S3
- Create on-chain commitment using the hash of the `vk.key` file
- Generate time-limited signed URLs for validators upon request

### Monitor Evaluation

- Watch validator requests: `pm2 logs miner`
- View metrics: https://wandb.ai/inferencelabs/omron
- View leaderboard: https://accelerator.omron.ai

## Troubleshooting

**Circuit Upload Fails**

- Verify storage credentials in .env
- Check network connectivity
- Verify all required files are present
- Check storage bucket permissions

**Validation Errors**

- Review validator logs for specific failures via WandB
- Verify input schema matches implementation

## For additional assistance

- Join Discord and reach out via the Subnet 2 channel: https://discord.gg/bittensor
- For security reports, please see our bug bounty program: https://immunefi.com/bug-bounty/omron/
- Otherwise, feel free to open a GitHub issue within the repository.
