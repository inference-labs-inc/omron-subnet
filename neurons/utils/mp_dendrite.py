import sys
import threading
import logging.handlers
import time
import asyncio
import aiohttp
import concurrent.futures
import os
from typing import Union, List

import bittensor as bt
from bittensor.core.settings import version_as_int

MAX_CONCURRENT_CONNECTIONS = int(os.getenv("MP_DENDRITE_MAX_CONNECTIONS", 64))


def silent_thread_hook(args):
    if isinstance(args.exc_value, EOFError):
        return
    sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)


threading.excepthook = silent_thread_hook


def safe_monitor(self):
    try:
        while True:
            try:
                record = self.dequeue(True)
            except EOFError:
                break
            except Exception:
                continue
            self.handle(record)
    except Exception:
        logging.exception("Exception occurred in safe_monitor")


if hasattr(logging.handlers, "QueueListener"):
    logging.handlers.QueueListener._monitor = safe_monitor


def get_endpoint_url(external_ip: str, target_axon: bt.AxonInfo, request_name: str):
    is_self = target_axon.ip in {str(external_ip), "127.0.0.1", "0.0.0.0", "localhost"}
    endpoint_ip = "127.0.0.1" if is_self else target_axon.ip
    return f"http://{endpoint_ip}:{str(target_axon.port)}/{request_name}"


def preprocess_synapse_for_request(
    ss58_address: str,
    nonce: int,
    uuid: str,
    external_ip: str,
    target_axon_info: bt.AxonInfo,
    synapse: bt.Synapse,
    timeout: float,
) -> bt.Synapse:
    synapse.timeout = timeout
    synapse.dendrite = bt.TerminalInfo(
        ip=external_ip,
        version=version_as_int,
        nonce=nonce,
        uuid=uuid,
        hotkey=ss58_address,
    )

    synapse.axon = bt.TerminalInfo(
        ip=target_axon_info.ip,
        port=target_axon_info.port,
        hotkey=target_axon_info.hotkey,
    )

    return synapse


def process_server_response(
    server_response: aiohttp.ClientResponse,
    json_response: dict,
    local_synapse: bt.Synapse,
):
    if server_response.status == 200:
        try:
            server_synapse = local_synapse.__class__(**json_response)
            for key in local_synapse.model_dump().keys():
                try:
                    setattr(local_synapse, key, getattr(server_synapse, key))
                except Exception:
                    pass
        except Exception as e:
            bt.logging.debug(f"Failed to create server synapse from response: {e}")
            # If we can't create the server synapse, try to extract individual fields
            for key, value in json_response.items():
                if hasattr(local_synapse, key):
                    try:
                        setattr(local_synapse, key, value)
                    except Exception:
                        pass
    else:
        if local_synapse.axon is None:
            local_synapse.axon = bt.TerminalInfo()
        local_synapse.axon.status_code = server_response.status
        local_synapse.axon.status_message = json_response.get("message")

    server_headers = bt.Synapse.from_headers(server_response.headers)

    local_synapse.dendrite.__dict__.update(
        {
            **local_synapse.dendrite.model_dump(exclude_none=True),
            **server_headers.dendrite.model_dump(exclude_none=True),
        }
    )

    local_synapse.axon.__dict__.update(
        {
            **local_synapse.axon.model_dump(exclude_none=True),
            **server_headers.axon.model_dump(exclude_none=True),
        }
    )

    local_synapse.dendrite.status_code = local_synapse.axon.status_code
    local_synapse.dendrite.status_message = local_synapse.axon.status_message


def process_error_message(
    synapse: bt.Synapse,
    request_name: str,
    exception: Exception,
) -> bt.Synapse:
    bt.logging.trace(f"Error in request {request_name}: {exception}")

    error_info = bt.core.dendrite.DENDRITE_ERROR_MAPPING.get(
        type(exception), bt.core.dendrite.DENDRITE_DEFAULT_ERROR
    )
    status_code, status_message = error_info

    if status_code:
        synapse.dendrite.status_code = status_code
    elif isinstance(exception, aiohttp.ClientResponseError):
        synapse.dendrite.status_code = str(exception.code)

    message = f"{status_message}: {str(exception)}"
    if isinstance(exception, aiohttp.ClientConnectorError):
        message = (
            f"{status_message} at {synapse.axon.ip}:{synapse.axon.port}/{request_name}"
        )
    elif isinstance(exception, asyncio.TimeoutError):
        message = f"{status_message} after {synapse.timeout} seconds"

    synapse.dendrite.status_message = message

    return synapse


async def call(
    ss58_address: str,
    nonce: int,
    signature: str,
    uuid: str,
    external_ip: str,
    session: aiohttp.ClientSession,
    target_axon: Union[bt.AxonInfo, bt.Axon],
    synapse_headers: dict,
    synapse_body: dict,
    timeout: float,
    synapse_class: type,
    request_name: str,
):
    start_time = time.time()
    target_axon = (
        target_axon.info() if isinstance(target_axon, bt.Axon) else target_axon
    )

    url = get_endpoint_url(external_ip, target_axon, request_name)

    synapse = synapse_class(**synapse_body).from_headers(synapse_headers)

    synapse = preprocess_synapse_for_request(
        ss58_address,
        nonce,
        uuid,
        external_ip,
        target_axon,
        synapse,
        timeout,
    )
    synapse.dendrite.signature = signature

    try:
        bt.logging.trace(
            f"dendrite | --> | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | 0 | Success"
        )
        async with session.post(
            url=url,
            headers=synapse.to_headers(),
            json=synapse_body,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            json_response = await response.json()
            process_server_response(response, json_response, synapse)

        synapse.dendrite.process_time = str(time.time() - start_time)

    except Exception as e:
        synapse = process_error_message(synapse, request_name, e)

    finally:
        # flake8: noqa
        bt.logging.trace(
            f"dendrite | <-- | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | {synapse.dendrite.status_code} | {synapse.dendrite.status_message}"
        )

        # Debug logging for response content
        if hasattr(synapse, "query_output") and synapse.query_output is None:
            bt.logging.debug(
                f"Warning: synapse.query_output is None for {synapse.name}"
            )

    return synapse


async def worker(
    ss58_address: str,
    nonce: int,
    uuid: str,
    external_ip: str,
    synapse_headers: dict,
    synapse_body: dict,
    axon_sig_pairs: list,
    timeout: float,
    synapse_class: type,
    request_name: str,
):
    chunk_size = len(axon_sig_pairs)
    connection_limit = min(MAX_CONCURRENT_CONNECTIONS, max(1, chunk_size))
    conn = aiohttp.TCPConnector(limit=connection_limit, limit_per_host=10)

    worker_timeout = timeout + 30

    try:
        result = await asyncio.wait_for(
            _worker_inner(
                ss58_address,
                nonce,
                uuid,
                external_ip,
                synapse_headers,
                synapse_body,
                axon_sig_pairs,
                timeout,
                synapse_class,
                request_name,
                conn,
            ),
            timeout=worker_timeout,
        )
        return result
    except asyncio.TimeoutError:
        bt.logging.error(f"Worker timed out after {worker_timeout}s")
        error_synapses = []
        for axon_dict, _ in axon_sig_pairs:
            error_synapse = synapse_class(**synapse_body).from_headers(synapse_headers)
            error_synapse.dendrite = bt.TerminalInfo(
                status_code="408", status_message="Worker timeout"
            )
            error_synapses.append(error_synapse)
        return error_synapses


async def _worker_inner(
    ss58_address: str,
    nonce: int,
    uuid: str,
    external_ip: str,
    synapse_headers: dict,
    synapse_body: dict,
    axon_sig_pairs: list,
    timeout: float,
    synapse_class: type,
    request_name: str,
    conn: aiohttp.TCPConnector,
):
    async with aiohttp.ClientSession(connector=conn) as sessions:
        return await asyncio.gather(
            *(
                call(
                    ss58_address=ss58_address,
                    nonce=nonce,
                    signature=signature,
                    uuid=uuid,
                    external_ip=external_ip,
                    session=sessions,
                    target_axon=bt.AxonInfo.from_parameter_dict(axon_dict),
                    synapse_headers=synapse_headers,
                    synapse_body=synapse_body,
                    timeout=timeout,
                    synapse_class=synapse_class,
                    request_name=request_name,
                )
                for axon_dict, signature in axon_sig_pairs
            )
        )


def run_chunk(
    ss58_address: str,
    nonce: int,
    uuid: str,
    external_ip: str,
    synapse_headers: dict,
    synapse_body: dict,
    axon_sig_pairs: list,
    timeout: float,
    synapse_class: type,
    request_name: str,
):
    try:
        chunk_timeout = timeout + 60

        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Chunk process exceeded {chunk_timeout}s timeout")

        # Set up timeout signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(chunk_timeout))

        try:
            result = asyncio.run(
                worker(
                    ss58_address,
                    nonce,
                    uuid,
                    external_ip,
                    synapse_headers,
                    synapse_body,
                    axon_sig_pairs,
                    timeout,
                    synapse_class,
                    request_name,
                )
            )
            return result
        finally:
            signal.alarm(0)
    except Exception as e:
        error_synapses = []
        for axon_dict, _ in axon_sig_pairs:
            try:
                error_synapse = synapse_class(**synapse_body).from_headers(
                    synapse_headers
                )
                error_synapse.dendrite = bt.TerminalInfo(
                    status_code="500", status_message=f"Process error: {str(e)}"
                )
                error_synapse.axon = bt.TerminalInfo(
                    ip=axon_dict.get("ip", "unknown"),
                    port=axon_dict.get("port", 0),
                    hotkey=axon_dict.get("hotkey", "unknown"),
                    status_code="500",
                    status_message=f"Process error: {str(e)}",
                )
                error_synapses.append(error_synapse)
            except Exception:
                error_synapse = synapse_class(**synapse_body)
                error_synapse.dendrite = bt.TerminalInfo(
                    status_code="500", status_message="Process execution failed"
                )
                error_synapses.append(error_synapse)
        return error_synapses


def sign(synapse: bt.Synapse, keypair: bt.Keypair):
    message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{synapse.axon.hotkey}.{synapse.dendrite.uuid}.{synapse.body_hash}"
    signature = f"0x{keypair.sign(message).hex()}"
    return signature


def chunkify(lst, n):
    k, m = divmod(len(lst), n)
    for i in range(n):
        yield lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]


def sign_axons(
    keypair: bt.Keypair,
    nonce: int,
    uuid: str,
    external_ip: str,
    axons: list[bt.AxonInfo],
    synapse: bt.Synapse,
    timeout: float,
):
    synapse = synapse.model_copy()
    for axon in axons:
        synapse = preprocess_synapse_for_request(
            keypair.ss58_address,
            nonce,
            uuid,
            external_ip,
            axon,
            synapse,
            timeout,
        )
        yield sign(synapse, keypair)


def mp_forward(
    keypair: bt.Keypair,
    uuid: str,
    external_ip: str,
    axons: list[bt.AxonInfo],
    synapse: bt.Synapse,
    timeout: float,
    nprocs: int = 8,
) -> list[bt.Synapse]:
    ss58_address = keypair.ss58_address
    synapse = synapse.model_copy()
    nonce = time.time_ns()
    request_name = synapse.__class__.__name__

    axon_dicts = [ax.to_parameter_dict() for ax in axons]
    signatures = list(
        sign_axons(keypair, nonce, uuid, external_ip, axons, synapse, timeout)
    )
    axon_sig_pairs = list(zip(axon_dicts, signatures))
    chunks = list(chunkify(axon_sig_pairs, nprocs))
    results = []

    with concurrent.futures.ProcessPoolExecutor(nprocs) as executor:
        chunk_futures = []
        for i, chunk in enumerate(chunks):
            future = executor.submit(
                run_chunk,
                ss58_address,
                nonce,
                uuid,
                external_ip,
                synapse.to_headers(),
                synapse.model_dump(),
                chunk,
                timeout,
                synapse.__class__,
                request_name,
            )
            chunk_futures.append((future, chunk))

        for future, chunk in chunk_futures:
            try:
                chunk_results = future.result(timeout=timeout + 90)
                results.extend(chunk_results)
            except Exception as e:
                for axon_dict, _ in chunk:
                    error_synapse = synapse.model_copy()
                    error_synapse.dendrite = bt.TerminalInfo(
                        status_code="500",
                        status_message=f"Process execution error: {str(e)}",
                    )
                    error_synapse.axon = bt.TerminalInfo(
                        ip=axon_dict.get("ip", "unknown"),
                        port=axon_dict.get("port", 0),
                        hotkey=axon_dict.get("hotkey", "unknown"),
                        status_code="500",
                        status_message=f"Process execution error: {str(e)}",
                    )
                    results.append(error_synapse)

    return results


class MultiprocessDendrite:
    def __init__(self, wallet: bt.wallet, external_ip: str = None, nprocs: int = 8):
        self.wallet = wallet
        self.external_ip = external_ip or "127.0.0.1"
        self.uuid = str(time.time_ns())
        self.nprocs = nprocs

    async def call(
        self,
        target_axon: Union[bt.AxonInfo, bt.Axon],
        synapse: bt.Synapse,
        timeout: float = 60.0,
        deserialize: bool = True,
        **kwargs,
    ) -> bt.Synapse:
        results = await self.forward(
            axons=[target_axon],
            synapse=synapse,
            timeout=timeout,
            deserialize=deserialize,
        )
        return results[0] if results else None

    async def forward(
        self,
        axons: Union[List[bt.AxonInfo], List[bt.Axon]],
        synapse: bt.Synapse,
        timeout: float = 60.0,
        deserialize: bool = True,
        **kwargs,
    ) -> List[bt.Synapse]:
        axon_infos = []
        for axon in axons:
            if isinstance(axon, bt.Axon):
                axon_infos.append(axon.info())
            else:
                axon_infos.append(axon)

        results = await asyncio.to_thread(
            mp_forward,
            keypair=self.wallet.hotkey,
            uuid=self.uuid,
            external_ip=self.external_ip,
            axons=axon_infos,
            synapse=synapse,
            timeout=timeout,
            nprocs=self.nprocs,
        )

        if deserialize:
            return [
                result.deserialize() if hasattr(result, "deserialize") else result
                for result in results
            ]
        return results
