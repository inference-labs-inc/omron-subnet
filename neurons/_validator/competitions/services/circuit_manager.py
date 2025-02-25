import os
import shutil
import json
import aiohttp
import bittensor as bt
from urllib.parse import urlparse
import traceback
from protocol import Competition


class CircuitManager:
    def __init__(self, temp_dir: str, competition_id: int, dendrite: bt.dendrite):
        self.temp_dir = temp_dir
        self.competition_id = competition_id
        self.dendrite = dendrite
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

    async def download_files(self, axon: bt.axon, hash: str, circuit_dir: str) -> bool:
        """Download circuit files from a miner."""
        try:
            bt.logging.debug(f"Requesting circuit files for hash {hash[:8]}...")
            bt.logging.debug(f"Circuit directory: {circuit_dir}")

            if not self.dendrite:
                bt.logging.error("Dendrite not initialized")
                return False

            synapse = Competition(
                id=self.competition_id,
                hash=hash,
                file_name="commitment",
            )

            bt.logging.debug(f"Sending request to axon: {axon.ip}:{axon.port}")
            response = await self.dendrite.forward(
                axons=[axon],
                synapse=synapse,
                timeout=60,
                deserialize=True,
            )

            if not response or not response[0]:
                bt.logging.error("No response from axon")
                return False

            response_data = response[0]
            if isinstance(response_data, dict):
                response_synapse = Competition(**response_data)
            else:
                response_synapse = response_data

            if not response_synapse.commitment:
                bt.logging.warning("No commitment data in response")
                return False

            try:
                commitment = json.loads(response_synapse.commitment)
                bt.logging.debug(f"Received commitment data: {commitment}")
            except json.JSONDecodeError:
                bt.logging.error("Invalid commitment data")
                return False

            if "signed_urls" not in commitment:
                bt.logging.error("No signed URLs in commitment data")
                return False

            signed_urls = commitment["signed_urls"]
            required_files = ["vk.key", "pk.key", "settings.json", "model.compiled"]
            all_files_downloaded = True

            # Create a new session for each download
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(force_close=True)
            ) as session:
                for file_name in required_files:
                    if file_name not in signed_urls:
                        bt.logging.error(f"Missing signed URL for {file_name}")
                        all_files_downloaded = False
                        break

                    url = signed_urls[file_name]
                    if not self._validate_url(url):
                        bt.logging.error(f"Invalid URL for {file_name}: {url}")
                        all_files_downloaded = False
                        break

                    file_path = os.path.join(circuit_dir, file_name)
                    try:
                        bt.logging.debug(f"Downloading {file_name} from {url}")
                        async with session.get(url, timeout=1200) as response:
                            response.raise_for_status()
                            content = await response.read()
                            bt.logging.debug(
                                f"Downloaded {len(content)} bytes for {file_name}"
                            )
                            with open(file_path, "wb") as f:
                                f.write(content)
                            bt.logging.debug(f"Saved {file_name} to {file_path}")
                    except Exception as e:
                        bt.logging.error(f"Failed to download {file_name}: {e}")
                        bt.logging.error(f"Stack trace: {traceback.format_exc()}")
                        all_files_downloaded = False
                        break

            if all_files_downloaded:
                bt.logging.success(f"Successfully downloaded all files for {hash[:8]}")
                bt.logging.debug("Final circuit directory contents:")
                for root, dirs, files in os.walk(circuit_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        bt.logging.debug(
                            f"- {file} ({os.path.getsize(file_path)} bytes)"
                        )
                return True
            else:
                bt.logging.error(f"Failed to download all files for {hash[:8]}")
                return False

        except Exception as e:
            bt.logging.error(f"Error downloading circuit files: {e}")
            bt.logging.error(f"Stack trace: {traceback.format_exc()}")
            return False
