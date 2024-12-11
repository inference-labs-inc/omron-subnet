from __future__ import annotations
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from jsonrpcserver import method, async_dispatch, Success, Error, InvalidParams
import bittensor as bt
from _validator.utils.proof_of_weights import (
    ProofOfWeightsItem,
)
import hashlib
from constants import MAX_SIGNATURE_LIFESPAN
from _validator.config import ValidatorConfig
import base64
import substrateinterface
import time
from OpenSSL import crypto
import datetime


class ValidatorKeysCache:
    """
    A class to cache validator keys. This is used to reduce the number of requests to the metagraph.
    """

    def __init__(self, config: ValidatorConfig) -> None:
        self.cached_keys: dict[int, list[str]] = {}
        self.cached_timestamps: dict[int, datetime.datetime] = {}
        self.config: ValidatorConfig = config

    def fetch_validator_keys(self, netuid: int) -> None:
        """
        Fetch the validator keys for a given netuid and cache them.
        """
        self.cached_keys[netuid] = [
            neuron.hotkey
            for neuron in self.config.subtensor.neurons_lite(netuid)
            if neuron.validator_permit
        ]
        self.cached_timestamps[netuid] = datetime.datetime.now() + datetime.timedelta(
            hours=12
        )

    def check_validator_key(self, ss58_address: str, netuid: int) -> bool:
        """
        Check if a given key is a validator key for a given netuid.
        """
        if ss58_address in self.config.api.whitelisted_public_keys:
            # If the sender is whitelisted, we don't need to check the key
            return True
        cache_timestamp = self.cached_timestamps.get(netuid, None)
        if cache_timestamp is None or cache_timestamp > datetime.datetime.now():
            self.fetch_validator_keys(netuid)
        return ss58_address in self.cached_keys.get(netuid, [])


class ValidatorAPI:
    """JSON-RPC WebSocket API for the Omron validator."""

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.app = FastAPI()
        self.external_requests_queue: list[(int, list[ProofOfWeightsItem])] = []
        self.active_connections: set[WebSocket] = set()
        self.validator_keys_cache = ValidatorKeysCache(config)

        if self.config.api.enabled:
            bt.logging.debug("Starting WebSocket API server...")
            self.setup_rpc_methods()
            self.ensure_valid_certificate()
            self.serve_axon()
            self.commit_cert_hash()
            bt.logging.success("WebSocket API server started")
        else:
            bt.logging.info(
                "API Disabled due to presence of `--ignore-external-requests` flag"
            )

    def ensure_valid_certificate(self):
        """Ensure the certificate is valid"""
        if not self.config.api.certificate_path:
            bt.logging.error(
                "No certificate path provided. "
                "Please provide a certificate path with `--certificate-path` or remove this flag."
            )
            return

        cert_path = self.config.api.certificate_path / "cert.pem"
        if not cert_path.exists():
            bt.logging.warning(
                "Certificate not found. A new self-signed SSL certificate will be issued."
            )
            self.issue_new_certificate()

    def issue_new_certificate(self):
        """Issue a new self-signed SSL certificate"""

        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 4096)

        cert = crypto.X509()
        cert.get_subject().CN = bt.axon(
            self.config.wallet, self.config.bt_config
        ).external_ip
        cert.set_serial_number(int(time.time()))
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(2 * 365 * 24 * 60 * 60)
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, "sha256")

        cert_path = self.config.api.certificate_path / "cert.pem"
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cert_path, "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))

        private_key_path = self.config.api.certificate_path / "key.pem"
        private_key_path.parent.mkdir(parents=True, exist_ok=True)
        with open(private_key_path, "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))

        bt.logging.success(
            f"New SSL certificate issued and saved to {cert_path} and {private_key_path}"
        )

    def setup_rpc_methods(self):
        """Initialize JSON-RPC method handlers"""

        @self.app.websocket("/rpc")
        async def websocket_endpoint(websocket: WebSocket):
            if (
                not self.config.api.verify_external_signatures
                and not await self.validate_connection(websocket.headers)
            ):
                raise HTTPException(
                    status_code=403, detail="Connection validation failed"
                )

            try:
                await websocket.accept()
                self.active_connections.add(websocket)

                async for data in websocket.iter_text():
                    response = await async_dispatch(data, methods=self.rpc_methods)
                    await websocket.send_text(str(response))

            except WebSocketDisconnect:
                bt.logging.debug("Client disconnected normally")
            except Exception as e:
                bt.logging.error(f"WebSocket error: {str(e)}")
            finally:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

        @method(name="omron.proof_of_weights")
        async def omron_proof_of_weights(
            websocket: WebSocket, params: dict[str, object]
        ) -> dict[str, object]:
            """Handle proof of weights request"""
            try:
                evaluation_data = params.get("evaluation_data")
                weights_version = params.get("weights_version")

                if not evaluation_data:
                    return InvalidParams(
                        data={"error": "Missing evaluation data"},
                    )

                if self.config.api.verify_external_signatures:
                    sender = websocket.headers["x-origin-ss58"]
                    if sender not in self.config.metagraph.hotkeys:
                        return Error(
                            code=403,
                            message="Sender not registered on origin subnet",
                            data={"sender": sender},
                        )

                    sender_id = self.config.metagraph.hotkeys.index(sender)
                    if not self.config.metagraph.validator_permit[sender_id]:
                        return Error(
                            code=403,
                            message="Sender lacks validator permit",
                            data={"sender": sender},
                        )

                self.external_requests_queue.insert(
                    0, (int(websocket.headers["x-netuid"]), evaluation_data)
                )

                proof = await self.wait_for_proof(evaluation_data, weights_version)

                return Success({"proof": proof["proof"], "weights": proof["weights"]})

            except Exception as e:
                bt.logging.error(f"Error processing request: {str(e)}")
                return Error(
                    code=500, message="Internal server error", data={"error": str(e)}
                )

        self.rpc_methods = [omron_proof_of_weights]

    def serve_axon(self):
        """Initialize and serve the Bittensor axon"""
        bt.logging.info(f"Serving axon on port {self.config.api.port}")
        axon = bt.axon(wallet=self.config.wallet, external_port=self.config.api.port)
        try:
            axon.serve(self.config.bt_config.netuid, self.config.subtensor)
            bt.logging.success("Axon served")
        except Exception as e:
            bt.logging.error(f"Error serving axon: {e}")

    async def stop(self):
        """Gracefully shutdown the WebSocket server"""
        for connection in self.active_connections:
            await connection.close()
        self.active_connections.clear()

    async def validate_connection(self, headers) -> bool:
        """Validate WebSocket connection request headers"""
        required_headers = ["x-timestamp", "x-origin-ss58", "x-signature", "x-netuid"]

        if not all(header in headers for header in required_headers):
            return False

        try:
            timestamp = int(headers["x-timestamp"])
            current_time = time.time()
            if current_time - timestamp > MAX_SIGNATURE_LIFESPAN:
                return False

            ss58_address = headers["x-origin-ss58"]
            signature = base64.b64decode(headers["x-signature"])
            netuid = int(headers["x-netuid"])

            public_key = substrateinterface.Keypair(ss58_address=ss58_address)
            if not public_key.verify(str(timestamp).encode(), signature):
                return False

            return self.validator_keys_cache.check_validator_key(ss58_address, netuid)

        except Exception as e:
            bt.logging.error(f"Validation error: {str(e)}")
            return False

    def commit_cert_hash(self):
        """Commit the cert hash to the chain. Clients will use this for certificate pinning."""

        existing_commitment = self.config.subtensor.get_commitment(
            self.config.subnet_uid, self.config.user_uid
        )

        if not self.config.api.certificate_path:
            return

        cert_path = self.config.api.certificate_path / "cert.pem"
        if not cert_path.exists():
            return

        with open(cert_path, "rb") as f:
            cert_hash = hashlib.sha256(f.read()).hexdigest()
            if cert_hash != existing_commitment:
                self.config.subtensor.commit(
                    self.config.wallet, self.config.subnet_uid, cert_hash
                )
            else:
                bt.logging.info("Certificate hash already committed to chain.")
