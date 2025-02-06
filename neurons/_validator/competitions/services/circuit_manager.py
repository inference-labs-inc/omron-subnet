import os
import shutil
import json
import asyncio
import bittensor as bt
from protocol import Competition
import hashlib
from urllib.parse import urlparse
import aiohttp


class CircuitManager:
    def __init__(self, temp_dir: str, competition_id: int):
        self.temp_dir = temp_dir
        self.competition_id = competition_id
        os.makedirs(temp_dir, exist_ok=True)
        self._loop = None

    def _get_event_loop(self):
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    async def _download_files_async(
        self, axon: bt.axon, hash: str, circuit_dir: str
    ) -> bool:
        dendrite = None
        try:
            dendrite = bt.dendrite()
            required_files = ["vk.key", "pk.key", "settings.json", "model.compiled"]

            synapse = Competition(
                id=self.competition_id, hash=hash, file_name="commitment"
            )
            response = await dendrite.call(target_axon=axon, synapse=synapse)
            response = Competition.model_validate(response)

            if not isinstance(response, Competition):
                bt.logging.error("Invalid response type from miner")
                return False

            if response.error:
                bt.logging.error(f"Error from miner: {response.error}")
                return False

            if response.commitment:
                commitment = json.loads(response.commitment)
                if "signed_urls" in commitment:
                    signed_urls = commitment["signed_urls"]

                    async with aiohttp.ClientSession() as session:
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
                                async with session.get(url) as response:
                                    response.raise_for_status()
                                    content = await response.read()
                                    with open(local_path, "wb") as f:
                                        f.write(content)
                                    bt.logging.debug(
                                        f"Downloaded {file_name} from signed URL"
                                    )

                                    if file_name == "vk.key":
                                        with open(local_path, "rb") as f:
                                            file_hash = hashlib.sha256(
                                                f.read()
                                            ).hexdigest()
                                        if file_hash != hash:
                                            bt.logging.error(
                                                f"Hash mismatch for vk.key: expected {hash}, got {file_hash}"
                                            )
                                            return False

                            except Exception as e:
                                bt.logging.error(f"Failed to download {file_name}: {e}")
                                return False
                        return True

            synapse = Competition(
                competition_id=self.competition_id,
                commitment_hash=hash,
                request_type="DOWNLOAD",
            )

            response = await axon.call(synapse)
            if not response or not response.success:
                bt.logging.error(f"Failed to download circuit files from {axon.hotkey}")
                return False

            for file_name, file_data in response.files.items():
                file_path = os.path.join(circuit_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(file_data)

            return True

        except Exception as e:
            bt.logging.error(f"Error downloading circuit files: {e}")
            return False
        finally:
            if dendrite:
                await dendrite.close()

    def download_files(self, axon: bt.axon, hash: str, circuit_dir: str) -> bool:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._download_files_async(axon, hash, circuit_dir), loop
                )
                return future.result()
            else:
                return loop.run_until_complete(
                    self._download_files_async(axon, hash, circuit_dir)
                )
        except Exception as e:
            bt.logging.error(f"Error downloading circuit files: {e}")
            return False

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
