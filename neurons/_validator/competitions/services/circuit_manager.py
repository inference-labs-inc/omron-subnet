import os
import shutil
import json
import aiohttp
import bittensor as bt
from urllib.parse import urlparse
import asyncio
from contextlib import asynccontextmanager
import traceback
from protocol import Competition


class CircuitManager:
    def __init__(self, temp_dir: str, competition_id: int, dendrite: bt.dendrite):
        self.temp_dir = temp_dir
        self.competition_id = competition_id
        self.dendrite = dendrite
        os.makedirs(temp_dir, exist_ok=True)
        self._session = None
        self._session_lock = asyncio.Lock()

    @asynccontextmanager
    async def _get_session(self):
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(force_close=True)
                )
            try:
                yield self._session
            except Exception as e:
                if self._session and not self._session.closed:
                    await self._session.close()
                self._session = None
                raise e

    async def close(self):
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None

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

            if not self.dendrite:
                bt.logging.error("Dendrite not initialized")
                return False

            synapse = Competition(
                id=self.competition_id,
                hash=hash,
                file_name="commitment",
            )

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
                bt.logging.error("No commitment data in response")
                return False

            try:
                commitment = json.loads(response_synapse.commitment)
            except json.JSONDecodeError:
                bt.logging.error("Invalid commitment data")
                return False

            if "signed_urls" not in commitment:
                bt.logging.error("No signed URLs in commitment data")
                return False

            signed_urls = commitment["signed_urls"]
            required_files = ["vk.key", "pk.key", "settings.json", "model.compiled"]
            all_files_downloaded = True

            async with self._get_session() as session:
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
                        async with session.get(url, timeout=60) as response:
                            response.raise_for_status()
                            content = await response.read()
                            with open(file_path, "wb") as f:
                                f.write(content)
                        bt.logging.debug(f"Downloaded {file_name} from signed URL")
                    except Exception as e:
                        bt.logging.error(f"Failed to download {file_name}: {e}")
                        all_files_downloaded = False
                        break

            if all_files_downloaded:
                bt.logging.success(f"Successfully downloaded all files for {hash[:8]}")
                return True
            else:
                bt.logging.error(f"Failed to download all files for {hash[:8]}")
                return False

        except Exception as e:
            bt.logging.error(f"Error downloading circuit files: {e}")
            traceback.print_exc()
            return False
