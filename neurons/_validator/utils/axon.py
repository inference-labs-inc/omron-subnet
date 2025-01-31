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


def process_server_response(
    server_response: "httpx.Response",
    json_response: dict,
    local_synapse: "bt.Synapse",
):
    # Check if the server responded with a successful status code
    if server_response.status_code == 200:
        # If the response is successful, overwrite local synapse state with
        # server's state only if the protocol allows mutation. To prevent overwrites,
        # the protocol must set Frozen = True
        server_synapse = local_synapse.__class__(**json_response)
        for key in local_synapse.model_dump().keys():
            try:
                # Set the attribute in the local synapse from the corresponding
                # attribute in the server synapse
                setattr(local_synapse, key, getattr(server_synapse, key))
            except Exception:
                # Ignore errors during attribute setting
                pass
    else:
        # If the server responded with an error, update the local synapse state
        if local_synapse.axon is None:
            local_synapse.axon = bt.TerminalInfo()
        local_synapse.axon.status_code = server_response.status_code
        local_synapse.axon.status_message = json_response.get("message")

    # Extract server headers and overwrite None values in local synapse headers
    server_headers = bt.Synapse.from_headers(dict(server_response.headers))  # type: ignore

    # Merge dendrite headers
    local_synapse.dendrite.__dict__.update(
        {
            **local_synapse.dendrite.model_dump(exclude_none=True),  # type: ignore
            **server_headers.dendrite.model_dump(exclude_none=True),  # type: ignore
        }
    )

    # Merge axon headers
    local_synapse.axon.__dict__.update(
        {
            **local_synapse.axon.model_dump(exclude_none=True),  # type: ignore
            **server_headers.axon.model_dump(exclude_none=True),  # type: ignore
        }
    )

    # Update the status code and status message of the dendrite to match the axon
    local_synapse.dendrite.status_code = local_synapse.axon.status_code  # type: ignore
    local_synapse.dendrite.status_message = local_synapse.axon.status_message  # type: ignore


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
            process_server_response(response, json_response, synapse)

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
