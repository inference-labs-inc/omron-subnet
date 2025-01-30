import traceback
import bittensor as bt
import time
import httpx
from typing import Union
from aiohttp.client_exceptions import InvalidUrlClientError

from constants import (
    VALIDATOR_REQUEST_TIMEOUT_SECONDS,
)
from _validator.core.request import Request


async def query_single_axon(dendrite: bt.dendrite, request: Request) -> Request | None:
    """
    Query a single axon with a request.

    Args:
        dendrite (bt.dendrite): The dendrite to use for querying.
        request (Request): The request to send.

    Returns:
        Request | None: The request with results populated, or None if the request failed.
    """

    try:
        result = await _call(
            dendrite=dendrite,
            target_axon=request.axon,
            synapse=request.synapse,
            timeout=VALIDATOR_REQUEST_TIMEOUT_SECONDS,
        )

        if not result:
            return None
        request.result = result
        request.response_time = (
            result.dendrite.process_time
            if result.dendrite.process_time is not None
            else VALIDATOR_REQUEST_TIMEOUT_SECONDS
        )

        request.deserialized = result.deserialize()
        return request

    except InvalidUrlClientError:
        bt.logging.warning(
            f"Ignoring UID as axon is not a valid URL: {request.uid}. {request.axon.ip}:{request.axon.port}"
        )
        return None

    except Exception as e:
        bt.logging.warning(f"Failed to query axon for UID: {request.uid}. Error: {e}")
        traceback.print_exc()
        return None


async def _call(
    dendrite: bt.dendrite,
    target_axon: Union["bt.AxonInfo", "bt.Axon"],
    synapse: "bt.Synapse" = bt.Synapse(),
    timeout: float = 12.0,
) -> "bt.Synapse":

    target_axon = (
        target_axon.info() if isinstance(target_axon, bt.Axon) else target_axon
    )

    # Build request endpoint from the synapse class
    request_name = synapse.__class__.__name__
    url = dendrite._get_endpoint_url(target_axon, request_name=request_name)

    # Preprocess synapse for making a request
    synapse = dendrite.preprocess_synapse_for_request(target_axon, synapse, timeout)

    try:
        # Log outgoing request
        dendrite._log_outgoing_request(synapse)

        # Make the HTTP POST request
        async with httpx.AsyncClient() as client:
            start_time = time.time()
            response = await client.post(
                url=url,
                headers=synapse.to_headers(),
                json=synapse.model_dump(),
                timeout=httpx.Timeout(timeout),
            )
            # Extract the JSON response from the server
            json_response = response.json()
            # Process the server response and fill synapse
            dendrite.process_server_response(response, json_response, synapse)

        # Set process time and log the response
        synapse.dendrite.process_time = str(time.time() - start_time)  # type: ignore

    except Exception as e:
        synapse = dendrite.process_error_message(synapse, request_name, e)

    finally:
        dendrite._log_incoming_response(synapse)

        # Log synapse event history
        dendrite.synapse_history.append(bt.Synapse.from_headers(synapse.to_headers()))

        # Return the updated synapse object after deserializing if requested
        return synapse
