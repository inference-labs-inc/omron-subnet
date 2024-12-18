import statistics
import threading
from typing import Optional
from wsgiref.simple_server import WSGIServer

from prometheus_client import Summary, start_http_server


_server: Optional[WSGIServer] = None
_thread: Optional[threading.Thread] = None

_validation_times: Optional[Summary] = None
_response_times: Optional[Summary] = None
_proof_sizes: Optional[Summary] = None
_verification_ratio: Optional[Summary] = None


def start_prometheus_logging(port: int) -> None:
    global _server, _thread
    _server, _thread = start_http_server(port)

    global _validation_times, _response_times, _proof_sizes, _verification_ratio
    _validation_times = Summary("validating_seconds", "Time spent validating responses")
    _response_times = Summary(
        "requests_seconds",
        "Time spent processing requests",
        ["aggregation_type", "model"],
    )
    _proof_sizes = Summary(
        "proof_sizes", "Size of proofs", ["aggregation_type", "model"]
    )
    _verification_ratio = Summary(
        "verified_proofs_ratio", "Verified proofs ratio", ["model"]
    )


def stop_prometheus_logging() -> None:
    global _server, _thread
    global _validation_times, _response_times, _proof_sizes, _verification_ratio
    if _server:
        _server.shutdown()
        _server = None
        _thread = None
        _validation_times = None
        _response_times = None
        _proof_sizes = None
        _verification_ratio = None


def log_validation_time(time: float) -> None:
    global _validation_times
    if _validation_times:
        _validation_times.observe(time)


def log_response_times(response_times: list[float], model_name: str) -> None:
    global _response_times
    if _response_times and response_times:
        _response_times.labels("max", model_name).observe(max(response_times))
        _response_times.labels("min", model_name).observe(min(response_times))
        _response_times.labels("mean", model_name).observe(
            statistics.mean(response_times)
        )
        _response_times.labels("median", model_name).observe(
            statistics.median(response_times)
        )


def log_proof_sizes(proof_sizes: list[int], model_name: str) -> None:
    global _proof_sizes
    if _proof_sizes and proof_sizes:
        _proof_sizes.labels("max", model_name).observe(max(proof_sizes))
        _proof_sizes.labels("min", model_name).observe(min(proof_sizes))
        _proof_sizes.labels("mean", model_name).observe(statistics.mean(proof_sizes))
        _proof_sizes.labels("median", model_name).observe(
            statistics.median(proof_sizes)
        )


def log_verification_ratio(value: float, model_name: str) -> None:
    global _verification_ratio
    if _verification_ratio:
        _verification_ratio.labels(model_name).observe(value)
