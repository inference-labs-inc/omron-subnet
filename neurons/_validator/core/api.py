import base64
import datetime
import json
import os
import traceback

import bittensor as bt
import substrateinterface
import uvicorn
from bittensor.axon import FastAPIThreadedServer
from fastapi import APIRouter, FastAPI
from fastapi.exceptions import HTTPException
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse

from _validator.config import ValidatorConfig
from _validator.utils.api import hash_inputs
from _validator.utils.proof_of_weights import (
    POW_DIRECTORY,
    POW_RECEIPT_DIRECTORY,
    ProofOfWeightsItem,
)
from constants import WHITELISTED_PUBLIC_KEYS


class ValidatorKeysCache:
    """
    A class to cache validator keys. This is used to reduce the number of requests to the Subtensor API.
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
        if (
            ss58_address in self.config.api.whitelisted_public_keys
            or ss58_address in WHITELISTED_PUBLIC_KEYS
        ):
            # if the sender is whitelisted, we don't need to check the key
            return True
        cache_timestamp = self.cached_timestamps.get(netuid, None)
        if cache_timestamp is None or cache_timestamp > datetime.datetime.now():
            self.fetch_validator_keys(netuid)
        return ss58_address in self.cached_keys.get(netuid, [])


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
        self.validator_keys_cache = ValidatorKeysCache(config)

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
            request_datetime_str: str = request.headers.get("X-Request-Datetime", "")
            encoded_signature: str = request.headers.get("X-Request-Signature", "")
            ss58_address = request.headers.get("X-ss58-Address", "")
            netuid = int(request.headers.get("X-Netuid", ""))

            # check if the request datetime header represents a valid datetime
            # it's important to check this because the datetime is used to calculate the signature
            request_datetime: datetime.datetime = datetime.datetime.strptime(
                request_datetime_str, "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            now = datetime.datetime.now(datetime.timezone.utc)
            if (
                request_datetime < (now - datetime.timedelta(minutes=1))
                or request_datetime > now
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Request timestamp is not within the last 1 minute.",
                )

            # construct the inputs to verify the signature
            inputs = f"{request.url}{request_datetime_str}{netuid}".encode()
            if request.method.upper() == "POST":
                inputs += await request.body()

            signature: bytes = base64.b64decode(encoded_signature)
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

        try:
            if not self.validator_keys_cache.check_validator_key(ss58_address, netuid):
                raise HTTPException(
                    status_code=403,
                    detail="Sender is not registered as a validator on the origin subnet.",
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
    ):
        """
        Fast API route handler to process external proof of weights requests.
        """
        await self.verify_request_signature(request=request)
        inputs = json.loads(await request.body())
        netuid = int(request.headers.get("X-Netuid", ""))

        try:
            input_hash = hash_inputs(inputs)

            self.external_requests_queue.insert(
                0,
                (
                    netuid,
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
