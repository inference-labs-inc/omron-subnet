"""
Script which allows users of any subnet to verify that validity of a proof of weights.
- Inputs:
    - inputs: An array of inputs for the reward function
    - proof: The raw proof JSON from the validator
    - netuid: The UID of the subnet being proven
"""

import argparse
import json
import os
import sys
import traceback

import bittensor as bt
import ezkl

netuid_to_model_hash = {
    2: "9af9ea2710f5a1ddcb22d953b6456b76c870d70335b2fcb03bcf011413ae53f6"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, required=True)
    parser.add_argument("--proof", type=str, required=True)
    parser.add_argument("--netuid", type=str, required=True)
    args = parser.parse_args()

    bt.logging.info(f"Verifying proof of weights for SN{args.netuid}")

    os.chdir(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "deployment_layer",
            f"model_{netuid_to_model_hash[args.netuid]}",
        )
    )

    with open("input.json", "w", encoding="utf-8") as f:
        content = json.load(f)
        content["input_data"] = args.inputs
        f.write(json.dumps(content))

    with open("proof.json", "w", encoding="utf-8") as f:
        f.write(args.proof)
    # Generate a witness using the given inputs
    ezkl.gen_witness()
    # Verify that the supplied proof and inputs are both valid
    try:
        ezkl.verify()
        bt.logging.success(f"Proof of weights for SN{args.netuid} is valid")
    except Exception:
        bt.logging.error(f"Proof of weights for SN{args.netuid} is invalid")
        traceback.print_exc()
        sys.exit(1)
