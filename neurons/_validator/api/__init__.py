from __future__ import annotations
import os
import traceback
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from jsonrpcserver import method, async_dispatch, Success, Error, InvalidParams
import bittensor as bt
from _validator.utils.proof_of_weights import ProofOfWeightsItem
import hashlib
from constants import MAX_SIGNATURE_LIFESPAN
from _validator.config import ValidatorConfig
import base64
import substrateinterface
import time
from _validator.api.cache import ValidatorKeysCache
import threading
import uvicorn
from _validator.api.certificate_manager import CertificateManager
from _validator.api.websocket_manager import WebSocketManager
from _validator.core.request_pipeline import RequestPipeline


class ValidatorAPI:
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.app = FastAPI()
        self.external_requests_queue: list[tuple[int, list[ProofOfWeightsItem]]] = []
        self.ws_manager = WebSocketManager()
        self.validator_keys_cache = ValidatorKeysCache(config)
        self.server_thread: threading.Thread | None = None
        self.request_pipeline: RequestPipeline | None = None
        self._setup_api()

    def set_request_pipeline(self, pipeline: RequestPipeline) -> None:
        """Set the request pipeline after initialization"""
        self.request_pipeline = pipeline

    def _setup_api(self) -> None:
        if not self.config.api.enabled:
            bt.logging.info("API Disabled: --ignore-external-requests flag present")
            return

        bt.logging.debug("Starting WebSocket API server...")
        if self.config.api.certificate_path:
            cert_manager = CertificateManager(self.config.api.certificate_path)
            cert_manager.ensure_valid_certificate(
                bt.axon(self.config.wallet).external_ip
            )
            self.commit_cert_hash()

        self.setup_rpc_methods()
        self.start_server()
        bt.logging.success("WebSocket API server started")

    def setup_rpc_methods(self) -> None:
        @self.app.websocket("/rpc")
        async def websocket_endpoint(websocket: WebSocket):
            if (
                self.config.api.verify_external_signatures
                and not await self.validate_connection(websocket.headers)
            ):
                raise HTTPException(
                    status_code=403, detail="Connection validation failed"
                )

            try:
                await self.ws_manager.connect(websocket)
                async for data in websocket.iter_text():
                    response = await async_dispatch(data, context=websocket)
                    await websocket.send_text(str(response))
            except WebSocketDisconnect:
                bt.logging.debug("Client disconnected normally")
            except Exception as e:
                bt.logging.error(f"WebSocket error: {str(e)}")
            finally:
                await self.ws_manager.disconnect(websocket)

        @method(name="omron.proof_of_weights")
        async def omron_proof_of_weights(
            websocket: WebSocket, **params: dict[str, object]
        ) -> dict[str, object]:
            evaluation_data = params.get("evaluation_data")
            _ = params.get("weights_version")

            if not evaluation_data:
                return InvalidParams(data={"error": "Missing evaluation data"})

            try:
                netuid = int(websocket.headers["x-netuid"])
                self.external_requests_queue.insert(0, (netuid, evaluation_data))
                return Success(data={"status": "Request received"})

            except Exception as e:
                bt.logging.error(f"Error processing request: {str(e)}")
                return Error(
                    code=500, message="Internal server error", data={"error": str(e)}
                )

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
        for connection in self.ws_manager.active_connections:
            await connection.close()
        self.ws_manager.active_connections.clear()

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
