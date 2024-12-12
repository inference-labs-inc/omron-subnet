from __future__ import annotations
import os
import traceback
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from jsonrpcserver import async_dispatch, Success, Error, InvalidParams
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
from _validator.api.cache import ValidatorKeysCache
import threading
import uvicorn


class ValidatorAPI:
    """JSON-RPC WebSocket API for the Omron validator."""

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.app = FastAPI()
        self.external_requests_queue: list[(int, list[ProofOfWeightsItem])] = []
        self.active_connections: set[WebSocket] = set()
        self.validator_keys_cache = ValidatorKeysCache(config)
        self.server_thread = None

        if self.config.api.enabled:
            bt.logging.debug("Starting WebSocket API server...")
            self.setup_rpc_methods()
            self.ensure_valid_certificate()
            self.start_server()
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

        cert_path = os.path.join(self.config.api.certificate_path, "cert.pem")
        if not os.path.exists(cert_path):
            bt.logging.warning(
                "Certificate not found. A new self-signed SSL certificate will be issued."
            )
            self.issue_new_certificate()

    def issue_new_certificate(self):
        """Issue a new self-signed SSL certificate"""

        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 4096)

        cert = crypto.X509()
        cert.get_subject().CN = bt.axon(self.config.wallet).external_ip
        cert.set_serial_number(int(time.time()))
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(2 * 365 * 24 * 60 * 60)
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, "sha256")

        cert_path = os.path.join(self.config.api.certificate_path, "cert.pem")
        os.makedirs(os.path.dirname(cert_path), exist_ok=True)
        with open(cert_path, "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))

        private_key_path = os.path.join(self.config.api.certificate_path, "key.pem")
        os.makedirs(os.path.dirname(private_key_path), exist_ok=True)
        with open(private_key_path, "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))

        bt.logging.success(
            f"New SSL certificate issued and saved to {cert_path} and {private_key_path}"
        )

    def setup_rpc_methods(self):
        """Initialize JSON-RPC method handlers"""

        async def omron_proof_of_weights(
            websocket: WebSocket, params: dict[str, object]
        ) -> dict[str, object]:
            """Handle proof of weights request"""
            try:
                evaluation_data = params.get("evaluation_data")
                weights_version = params.get("weights_version")

                if not evaluation_data:
                    return InvalidParams(data={"error": "Missing evaluation data"})

                self.external_requests_queue.insert(
                    0, (int(websocket.headers["x-netuid"]), evaluation_data)
                )

                proof = await self.wait_for_proof(evaluation_data, weights_version)

                return Success({"proof": proof["proof"], "weights": proof["weights"]})

            except Exception as e:
                traceback.print_exc()
                bt.logging.error(f"Error processing request: {str(e)}")
                return Error(
                    code=500, message="Internal server error", data={"error": str(e)}
                )

        @self.app.websocket("/rpc")
        async def websocket_endpoint(websocket: WebSocket):
            try:
                if self.config.api.verify_external_signatures:
                    if not await self.validate_connection(websocket.headers):
                        raise HTTPException(
                            status_code=403, detail="Connection validation failed"
                        )

                await websocket.accept()
                self.active_connections.add(websocket)

                async for data in websocket.iter_text():
                    methods = {
                        "omron.proof_of_weights": lambda params: omron_proof_of_weights(
                            websocket, params
                        )
                    }
                    response = await async_dispatch(data, methods=methods)
                    await websocket.send_text(str(response))

            except WebSocketDisconnect:
                bt.logging.debug("Client disconnected normally")
            except Exception as e:
                bt.logging.error(f"WebSocket error: {str(e)}")
            finally:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

    def start_server(self):
        """Start the uvicorn server in a separate thread"""
        self.server_thread = threading.Thread(
            target=uvicorn.run,
            args=(self.app,),
            kwargs={
                "host": "0.0.0.0",
                "port": self.config.api.port,
                "ssl_keyfile": os.path.join(
                    self.config.api.certificate_path, "key.pem"
                ),
                "ssl_certfile": os.path.join(
                    self.config.api.certificate_path, "cert.pem"
                ),
            },
            daemon=True,
        )
        self.server_thread.start()
        try:
            bt.logging.info(f"Serving axon on port {self.config.api.port}")
            axon = bt.axon(
                wallet=self.config.wallet, external_port=self.config.api.port
            )
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

        existing_commitment = None
        try:
            existing_commitment = self.config.subtensor.get_commitment(
                self.config.subnet_uid, self.config.user_uid
            )
        except Exception:
            bt.logging.warning(
                "Error getting existing commitment. Assuming no commitment exists."
            )
            traceback.print_exc()

        if not self.config.api.certificate_path:
            return

        cert_path = os.path.join(self.config.api.certificate_path, "cert.pem")
        if not os.path.exists(cert_path):
            return

        with open(cert_path, "rb") as f:
            cert_hash = hashlib.sha256(f.read()).hexdigest()
            if cert_hash != existing_commitment:
                try:
                    self.config.subtensor.commit(
                        self.config.wallet, self.config.subnet_uid, cert_hash
                    )
                    bt.logging.success("Certificate hash committed to chain.")
                except Exception as e:
                    bt.logging.error(f"Error committing certificate hash: {str(e)}")
                    traceback.print_exc()
            else:
                bt.logging.info("Certificate hash already committed to chain.")
