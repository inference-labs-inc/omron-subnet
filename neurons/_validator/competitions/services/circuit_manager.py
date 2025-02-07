import os
import shutil
import json
import aiohttp
import bittensor as bt
from urllib.parse import urlparse
import asyncio
from contextlib import asynccontextmanager
import base64
import traceback


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

            request = {"type": "circuit_request", "hash": hash}

            if not self.dendrite:
                bt.logging.error("Dendrite not initialized")
                return False

            response = await self.dendrite.forward(
                axons=[axon],
                synapse=request,
                timeout=60,
                deserialize=True,
                streaming=False,
            )

            if not response or not response[0]:
                bt.logging.error(f"No response from axon for hash {hash[:8]}")
                return False

            response = response[0]
            bt.logging.debug(f"Got response for hash {hash[:8]}, saving files...")

            expected_files = ["circuit.json", "circuit.wasm", "circuit.zkey", "vk.key"]

            for file in expected_files:
                if file not in response:
                    bt.logging.error(f"Missing {file} in response")
                    return False

                file_path = os.path.join(circuit_dir, file)
                try:
                    if file.endswith(".json"):
                        with open(file_path, "w") as f:
                            json.dump(response[file], f)
                    else:
                        with open(file_path, "wb") as f:
                            f.write(base64.b64decode(response[file]))
                    bt.logging.debug(f"Saved {file}")
                except Exception as e:
                    bt.logging.error(f"Failed to save {file}: {e}")
                    return False

            bt.logging.success(f"Successfully downloaded all files for {hash[:8]}")
            return True

        except Exception as e:
            bt.logging.error(f"Error downloading circuit files: {e}")
            traceback.print_exc()
            return False
