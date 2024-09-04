import asyncio
import random
import traceback
import bittensor as bt

from constants import (
    MAX_CONCURRENT_REQUESTS,
    VALIDATOR_REQUEST_TIMEOUT_SECONDS,
    VALIDATOR_AGG_REQUEST_TIMEOUT_SECONDS,
)


async def query_axons(dendrite, requests):
    bt.logging.trace("Querying axons")
    random.shuffle(requests)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def send_request(request):
        async with semaphore:
            axon = request["axon"]
            return await dendrite.forward(
                axons=[axon],
                synapse=request["synapse"],
                timeout=(
                    VALIDATOR_REQUEST_TIMEOUT_SECONDS
                    if not request["aggregation"]
                    else VALIDATOR_AGG_REQUEST_TIMEOUT_SECONDS
                ),
                deserialize=False,
            )

    tasks = [send_request(request) for request in requests]

    try:
        results = await asyncio.gather(*tasks)
        for i, sublist in enumerate(results):
            result = sublist[0]
            try:
                requests[i].update(
                    {
                        "result": result,
                        "response_time": (
                            result.dendrite.process_time
                            if result.dendrite.process_time is not None
                            else (
                                VALIDATOR_REQUEST_TIMEOUT_SECONDS
                                if not requests[i]["aggregation"]
                                else VALIDATOR_AGG_REQUEST_TIMEOUT_SECONDS
                            )
                        ),
                        "deserialized": result.deserialize(),
                    }
                )
            except Exception as e:
                bt.logging.warning(
                    f"""Failed to add result, response time and deserialized output to request
                        for UID: {requests[i]['uid']}. Error: {e}"""
                )
                traceback.print_exc()
                requests[i].update(
                    {
                        "result": result,
                        "response_time": (
                            VALIDATOR_REQUEST_TIMEOUT_SECONDS
                            if not requests[i]["aggregation"]
                            else VALIDATOR_AGG_REQUEST_TIMEOUT_SECONDS
                        ),
                        "deserialized": None,
                    }
                )
        requests.sort(key=lambda x: x["uid"])
        return requests
    except Exception as e:
        bt.logging.exception("Error while querying axons.\nReport this error: ", e)
        traceback.print_exc()
        raise e
