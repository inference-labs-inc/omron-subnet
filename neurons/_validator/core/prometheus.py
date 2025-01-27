import statistics
import threading
from typing import Optional
from wsgiref.simple_server import WSGIServer

from prometheus_client import Histogram, Gauge, Counter, start_http_server


_server: Optional[WSGIServer] = None
_thread: Optional[threading.Thread] = None

# Performance Metrics
_validation_times: Optional[Histogram] = None
_response_times: Optional[Histogram] = None
_proof_sizes: Optional[Histogram] = None

# Success/Failure Metrics
_verification_ratio: Optional[Histogram] = None
_verification_failures: Optional[Counter] = None
_timeout_counter: Optional[Counter] = None
_network_errors: Optional[Counter] = None

# Resource Usage
_active_requests: Optional[Gauge] = None
_processed_uids: Optional[Gauge] = None
_memory_usage: Optional[Gauge] = None

# Business Metrics
_total_proofs_verified: Optional[Counter] = None
_total_requests_processed: Optional[Counter] = None
_avg_response_time: Optional[Gauge] = None


def start_prometheus_logging(port: int) -> None:
    global _server
    global _thread
    global _validation_times
    global _response_times
    global _proof_sizes
    global _verification_ratio
    global _verification_failures
    global _timeout_counter
    global _network_errors
    global _active_requests
    global _processed_uids
    global _memory_usage
    global _total_proofs_verified
    global _total_requests_processed
    global _avg_response_time

    _server, _thread = start_http_server(port)

    # Performance Metrics
    _validation_times = Histogram(
        "validating_seconds",
        "Time spent validating responses",
        buckets=(
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ),
    )
    _response_times = Histogram(
        "requests_seconds",
        "Time spent processing requests",
        ["aggregation_type", "model"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0),
    )
    _proof_sizes = Histogram(
        "proof_sizes",
        "Size of proofs in bytes",
        ["aggregation_type", "model"],
        buckets=(1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000),
    )

    # Success/Failure Metrics
    _verification_ratio = Histogram(
        "verified_proofs_ratio", "Ratio of successfully verified proofs", ["model"]
    )
    _verification_failures = Counter(
        "verification_failures_total",
        "Total number of proof verification failures",
        ["model", "failure_type"],
    )
    _timeout_counter = Counter(
        "timeouts_total", "Total number of request timeouts", ["model"]
    )
    _network_errors = Counter(
        "network_errors_total", "Total number of network errors", ["error_type"]
    )

    # Resource Usage
    _active_requests = Gauge("active_requests", "Number of currently active requests")
    _processed_uids = Gauge("processed_uids", "Number of processed UIDs")
    _memory_usage = Gauge("memory_usage_bytes", "Current memory usage in bytes")

    # Business Metrics
    _total_proofs_verified = Counter(
        "total_proofs_verified",
        "Total number of proofs successfully verified",
        ["model"],
    )
    _total_requests_processed = Counter(
        "total_requests_processed",
        "Total number of requests processed",
        ["model", "status"],
    )
    _avg_response_time = Gauge(
        "avg_response_time_seconds", "Moving average of response times", ["model"]
    )


def stop_prometheus_logging() -> None:
    global _server
    global _thread
    global _validation_times
    global _response_times
    global _proof_sizes
    global _verification_ratio
    global _verification_failures
    global _timeout_counter
    global _network_errors
    global _active_requests
    global _processed_uids
    global _memory_usage
    global _total_proofs_verified
    global _total_requests_processed
    global _avg_response_time

    if _server:
        _server.shutdown()
        _server = None
        _thread = None
        _validation_times = None
        _response_times = None
        _proof_sizes = None
        _verification_ratio = None
        _verification_failures = None
        _timeout_counter = None
        _network_errors = None
        _active_requests = None
        _processed_uids = None
        _memory_usage = None
        _total_proofs_verified = None
        _total_requests_processed = None
        _avg_response_time = None


def log_validation_time(time: float) -> None:
    if _validation_times:
        _validation_times.observe(time)


def log_response_times(response_times: list[float], model_name: str) -> None:
    if _response_times and response_times:
        _response_times.labels("max", model_name).observe(max(response_times))
        _response_times.labels("min", model_name).observe(min(response_times))
        mean = statistics.mean(response_times)
        _response_times.labels("mean", model_name).observe(mean)
        _response_times.labels("median", model_name).observe(
            statistics.median(response_times)
        )

        if _avg_response_time:
            _avg_response_time.labels(model_name).set(mean)

        if _total_requests_processed:
            _total_requests_processed.labels(model_name, "success").inc(
                len(response_times)
            )


def log_proof_sizes(proof_sizes: list[int], model_name: str) -> None:
    if _proof_sizes and proof_sizes:
        _proof_sizes.labels("max", model_name).observe(max(proof_sizes))
        _proof_sizes.labels("min", model_name).observe(min(proof_sizes))
        _proof_sizes.labels("mean", model_name).observe(statistics.mean(proof_sizes))
        _proof_sizes.labels("median", model_name).observe(
            statistics.median(proof_sizes)
        )


def log_verification_ratio(value: float, model_name: str) -> None:
    if _verification_ratio:
        _verification_ratio.labels(model_name).observe(value)

    if _total_proofs_verified and value > 0:
        _total_proofs_verified.labels(model_name).inc()


def log_verification_failure(model_name: str, failure_type: str) -> None:
    if _verification_failures:
        _verification_failures.labels(model_name, failure_type).inc()

    if _total_requests_processed:
        _total_requests_processed.labels(model_name, "failed").inc()


def log_timeout(model_name: str) -> None:
    if _timeout_counter:
        _timeout_counter.labels(model_name).inc()

    if _total_requests_processed:
        _total_requests_processed.labels(model_name, "timeout").inc()


def log_network_error(error_type: str) -> None:
    if _network_errors:
        _network_errors.labels(error_type).inc()


def log_request_metrics(
    active_requests: int,
    processed_uids: int,
    memory_bytes: Optional[int] = None,
) -> None:
    if _active_requests:
        _active_requests.set(active_requests)
    if _processed_uids:
        _processed_uids.set(processed_uids)
    if _memory_usage and memory_bytes:
        _memory_usage.set(memory_bytes)
