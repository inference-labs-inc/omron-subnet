import uvicorn
from bittensor.axon import FastAPIThreadedServer
import traceback
from fastapi import APIRouter, FastAPI
import bittensor as bt
from _validator.utils.proof_of_weights import (
    POW_DIRECTORY,
    POW_RECEIPT_DIRECTORY,
    ProofOfWeightsItem,
)
from _validator.utils.api import hash_inputs
from _validator.models.api import PowInputModel
from _validator.config import ValidatorConfig
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.requests import Request
from constants import WHITELISTED_PUBLIC_KEYS
import base64
import json
import os
import substrateinterface


class ValidatorAPI:
    """
    API for the validator.
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.app = FastAPI()
        log_level = "trace" if bt.logging.__trace_on__ else "critical"
        self.fast_config = uvicorn.Config(
            self.app,
            host=self.config.api.host,
            port=self.config.api.port,
            log_level=log_level,
            workers=self.config.api.workers,
        )
        self.fast_server = FastAPIThreadedServer(config=self.fast_config)

        self.router = APIRouter()
        self.app.include_router(self.router)
        self.api_server_started = False
        self.external_requests_queue: list[(int, list[ProofOfWeightsItem])] = []

        if self.config.api.enabled:
            bt.logging.debug("Starting API server...")
            self.start_api_server()
            self.serve_axon()
            bt.logging.success("API server started")
        else:
            bt.logging.info(
                "API Disabled due to presence of `--ignore-external-requests` flag"
            )

    def serve_axon(self):
        bt.logging.info(f"Serving axon on port {self.config.api.port}")
        axon = bt.axon(wallet=self.config.wallet, external_port=self.config.api.port)
        try:
            axon.serve(self.config.bt_config.netuid, self.config.subtensor)
            bt.logging.success("Axon served")
        except Exception as e:
            bt.logging.error(f"Error serving axon: {e}")

    async def verify_request_signature(self, request: Request):
        """
        Verify the signature of an incoming request.
        """
        if not self.config.api.verify_external_signatures:
            return

        try:
            salt: str = request.headers.get("X-Salt", "")
            inputs = f"{request.url}{salt}".encode()
            if request.method.upper() == "POST":
                inputs += await request.body()
            encoded_signature: str = request.headers.get("X-Request-Signature", "")
            signature: bytes = base64.b64decode(encoded_signature)
            ss58_address = request.headers.get("X-ss58-Address", "")
            public_key = substrateinterface.Keypair(ss58_address=ss58_address)
        except Exception:
            bt.logging.error(
                f"Failed to verify incoming request body. {traceback.format_exc()}"
            )
            raise HTTPException(
                status_code=400,
                detail="Request failed validation.",
            )

        if not public_key.verify(data=inputs, signature=signature):
            raise HTTPException(
                status_code=401,
                detail="Signature verification failed.",
            )

        if (
            ss58_address in self.config.api.whitelisted_public_keys
            or ss58_address in WHITELISTED_PUBLIC_KEYS
        ):
            # if the sender is whitelisted, we don't need to check the key
            return

        try:
            if ss58_address not in self.config.metagraph.hotkeys:
                raise HTTPException(
                    status_code=403,
                    detail="Sender is not registered on the origin subnet.",
                )
            sender_id = self.config.metagraph.hotkeys.index(ss58_address)
            if not self.config.metagraph.validator_permit[sender_id]:
                raise HTTPException(
                    status_code=403,
                    detail="Sender does not have a validator permit on the origin subnet.",
                )
        except HTTPException as e:
            raise e
        except Exception:
            bt.logging.error(
                f"Unexpected error validating sender: {traceback.format_exc()}"
            )
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while validating the sender.",
            )

    async def process_external_request(
        self,
        request: Request,
        data: PowInputModel,
    ):
        """
        Fast API route handler to process external proof of weights requests.
        """
        await self.verify_request_signature(request=request)

        try:
            inputs = data.inputs
            input_hash = hash_inputs(inputs)

            self.external_requests_queue.insert(
                0,
                (
                    data.netuid,
                    inputs,
                ),
            )
        except Exception:
            bt.logging.error(
                f"Error processing external request: {traceback.format_exc()}"
            )
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while processing the request.",
            )
        bt.logging.success(
            f"Received external request for {input_hash}. Queue is now at {len(self.external_requests_queue)} items."
        )

        return JSONResponse(content={"hash": input_hash})

    async def get_proof_of_weights(self, input_hash: str, request: Request):
        """
        Fast API route handler to get proof of weights file for a given input hash.
        """
        await self.verify_request_signature(request=request)
        filename = f"{input_hash}.json"
        filepath = os.path.join(POW_DIRECTORY, filename)
        if os.path.exists(filepath):
            return FileResponse(
                path=filepath, filename=filename, media_type="application/json"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="The requested proof could not be found.",
            )

    async def get_receipt(self, transaction_hash: str, request: Request):
        """
        Fast API route handler to get receipt file for a given transaction hash.
        """
        await self.verify_request_signature(request=request)
        filepath = os.path.join(POW_RECEIPT_DIRECTORY, transaction_hash)
        if os.path.exists(filepath):
            return FileResponse(
                path=filepath, filename=transaction_hash, media_type="application/json"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="Receipt file not found",
            )

    def index(self):
        return JSONResponse(content={"message": "Validator API enabled"})

    def start_api_server(self):
        """
        Start the server.
        """
        self.router.add_api_route(
            "/",
            endpoint=self.index,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/submit-inputs",
            endpoint=self.process_external_request,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/get-proof-of-weights",
            endpoint=self.get_proof_of_weights,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/receipts",
            endpoint=self.get_receipt,
            methods=["GET"],
        )
        self.app.include_router(self.router)

        self.fast_server.start()
        self.api_server_started = True

    def stop(self):
        """
        Stop the server.
        """
        if self.api_server_started:
            self.fast_server.stop()
            self.api_server_started = False
