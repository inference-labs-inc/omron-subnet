from __future__ import annotations
from rich.console import Console, JustifyMethod
from rich.table import Table
import utils.wandb_logger as wandb_logger
import torch
from _validator.models.miner_response import MinerResponse


def create_and_print_table(
    title: str, columns: list[tuple[str, JustifyMethod, str]], rows: list[list[str]]
):
    """
    Create and print a table.

    Args:
        title (str): The title of the table.
        columns (list[tuple[str, JustifyMethod, str]]): A list of tuples containing column information.
            Each tuple should contain (column_name, justification, style).
        rows (list[list[str]]): A list of rows, where each row is a list of string values.

    """
    table = Table(title=title)
    for col_name, justify, style in columns:
        table.add_column(col_name, justify=justify, style=style, no_wrap=True)
    for row in rows:
        table.add_row(*row)
    console = Console()
    console.width = 120
    console.print(table)


def log_tensor_data(title: str, data: torch.Tensor, log_key: str):
    """
    Log tensor data to a table and Weights & Biases.

    Args:
        title (str): The title of the table.
        data (torch.Tensor): The tensor data to be logged.
        log_key (str): The key used for logging in Weights & Biases.
    """
    rows = [[str(uid), str(round(value.item(), 4))] for uid, value in enumerate(data)]
    create_and_print_table(
        title, [("uid", "right", "cyan"), (log_key, "right", "magenta")], rows
    )
    wandb_logger.safe_log(
        {log_key: {uid: value.item() for uid, value in enumerate(data)}}
    )


def log_scores(scores: torch.Tensor):
    """
    Log scores to a table and Weights & Biases.

    Args:
        scores (torch.Tensor): The scores tensor to be logged.

    """
    log_tensor_data("scores", scores, "scores")


def log_weights(weights: torch.Tensor):
    """
    Log weights to a table and Weights & Biases.

    Args:
        weights (torch.Tensor): The weights tensor to be logged.
    """
    log_tensor_data("weights", weights, "weights")


def log_verify_result(results: list[tuple[int, bool]]):
    """
    Log verification results to a table and Weights & Biases.

    Args:
        results (list[tuple[int, bool]]): A list of tuples containing (uid, verification_result).

    """
    rows = [[str(uid), str(result)] for uid, result in results]
    create_and_print_table(
        "proof verification result",
        [("uid", "right", "cyan"), ("Verified?", "right", "magenta")],
        rows,
    )
    wandb_logger.safe_log(
        {"verification_results": {uid: int(result) for uid, result in results}}
    )


def log_responses(responses: list[MinerResponse]):
    """
    Log miner responses to a table and Weights & Biases.

    Args:
        responses (list[MinerResponse]): A list of MinerResponse objects to be logged.
    """
    columns = [
        ("UID", "right", "magenta"),
        ("Verification Result", "right", "magenta"),
        ("Response Time", "right", "magenta"),
        ("Proof Size", "right", "magenta"),
        (
            "Circuit Name",
            "left",
            "magenta",
        ),
        ("Proof System", "left", "magenta"),
    ]

    sorted_responses = sorted(responses, key=lambda x: x.uid)

    circuit = sorted_responses[0].circuit if len(sorted_responses) > 0 else None
    circuit_name = circuit.metadata.name if circuit is not None else "Unknown"
    proof_system = circuit.metadata.proof_system if circuit is not None else "Unknown"

    rows = [
        [
            str(response.uid),
            str(response.verification_result),
            str(response.response_time),
            str(response.proof_size),
            circuit_name,
            proof_system,
        ]
        for response in sorted_responses
    ]
    create_and_print_table("Responses", columns, rows)

    wandb_log = {"responses": {}}
    for response in sorted_responses:
        if response.uid not in wandb_log["responses"]:
            wandb_log["responses"][response.uid] = {}
        wandb_log["responses"][response.uid][circuit_name] = {
            "verification_result": int(response.verification_result),
            "response_time": response.response_time,
            "proof_size": response.proof_size,
        }
    wandb_logger.safe_log(wandb_log)


def log_system_metrics(response_times, verified_count, circuit):

    if response_times:
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        mean_response_time = sum(response_times) / len(response_times)
        median_response_time = sorted(response_times)[len(response_times) // 2]
        wandb_logger.safe_log(
            {
                f"{circuit}": {
                    "max_response_time": max_response_time,
                    "min_response_time": min_response_time,
                    "mean_response_time": mean_response_time,
                    "median_response_time": median_response_time,
                    "total_responses": len(response_times),
                    "verified_responses": verified_count,
                }
            }
        )
