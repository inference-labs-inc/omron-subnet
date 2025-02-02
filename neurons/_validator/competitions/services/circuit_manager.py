import os
import shutil
import json
import requests
import bittensor as bt
from protocol import Competition
from urllib.parse import urlparse


class CircuitManager:
    def __init__(self, temp_dir: str, competition_id: int):
        self.temp_dir = temp_dir
        self.competition_id = competition_id
        os.makedirs(temp_dir, exist_ok=True)

    def cleanup_temp_files(self, circuit_dir: str):
        try:
            if os.path.exists(circuit_dir):
                shutil.rmtree(circuit_dir)
            temp_files = [
                os.path.join("/tmp/omron", f)
                for f in ["temp_proof.json", "temp_witness.json"]
            ]
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
        except Exception as e:
            bt.logging.warning(f"Error cleaning up temp files: {e}")

    def _validate_url(self, url: str) -> bool:
        """Validate that URL is from R2 or S3"""
        try:
            parsed = urlparse(url)
            return any(
                domain in parsed.netloc
                for domain in [
                    "r2.cloudflarestorage.com",
                    "s3.amazonaws.com",
                    ".r2.dev",
                ]
            )
        except Exception:
            return False

    def download_files(self, axon: bt.axon, hash: str, circuit_dir: str) -> bool:
        try:
            dendrite = bt.dendrite()
            required_files = ["vk.key", "pk.key", "settings.json", "model.compiled"]

            synapse = Competition(
                id=self.competition_id, hash=hash, file_name="commitment"
            )
            response = dendrite.query(axons=[axon], synapse=synapse)[0]

            if response.error:
                bt.logging.error(f"Error from miner: {response.error}")
                return False

            if not response.commitment:
                bt.logging.error("No commitment data received from miner")
                return False

            commitment = json.loads(response.commitment)
            if "signed_urls" not in commitment:
                bt.logging.error("No signed URLs in commitment data")
                return False

            signed_urls = commitment["signed_urls"]

            for file_name in required_files:
                if file_name not in signed_urls:
                    bt.logging.error(f"Missing signed URL for {file_name}")
                    return False

                url = signed_urls[file_name]
                if not self._validate_url(url):
                    bt.logging.error(f"Invalid URL for {file_name}: {url}")
                    return False

                local_path = os.path.join(circuit_dir, file_name)
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(local_path, "wb") as f:
                        f.write(response.content)
                    bt.logging.debug(f"Downloaded {file_name} from signed URL")
                except Exception as e:
                    bt.logging.error(f"Failed to download {file_name}: {e}")
                    return False

            return True
        except Exception as e:
            bt.logging.error(f"Error downloading circuit files: {e}")
            return False
