import torch
import json
import random as rand


PRICE_SCORE_WEIGHT = 0.86
DATE_SCORE_WEIGHT = 0.14


class Circuit(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        actual_price: torch.Tensor,
        predicted_price: torch.Tensor,
        date_difference: torch.Tensor,
        price_score_weight: torch.Tensor,
        date_score_weight: torch.Tensor,
    ):
        price_score_weight = price_score_weight.view(-1, 1).expand_as(actual_price)
        date_score_weight = date_score_weight.view(-1, 1).expand_as(actual_price)

        date_score = torch.mul(
            torch.div(
                torch.max(torch.tensor(0), torch.sub(14, date_difference)),
                torch.tensor(14),
            ),
            torch.tensor(100),
        )

        price_difference = torch.div(
            torch.abs(torch.sub(actual_price, predicted_price)), actual_price
        )
        price_score = torch.max(
            torch.tensor(0),
            torch.sub(
                torch.tensor(100), torch.mul(price_difference, torch.tensor(100))
            ),
        )

        final_score = torch.add(
            torch.mul(price_score, price_score_weight),
            torch.mul(date_score, date_score_weight),
        )

        return final_score


if __name__ == "__main__":
    circuit = Circuit()
    batch_size = 300
    actual_prices = torch.tensor(
        [[float(rand.randint(200000, 2000000))] for _ in range(batch_size)]
    )
    predicted_prices = torch.tensor(
        [[p[0] * (1 + (rand.random() - 0.5) * 0.2)] for p in actual_prices]
    )
    date_differences = torch.tensor([[rand.randint(0, 21)] for _ in range(batch_size)])
    dummy_input = {
        "actual_price": actual_prices,
        "predicted_price": predicted_prices,
        "date_difference": date_differences,
        "price_score_weight": torch.tensor([PRICE_SCORE_WEIGHT]),
        "date_score_weight": torch.tensor([DATE_SCORE_WEIGHT]),
    }

    input_data = {
        "input_data": [
            [float(p[0]) for p in actual_prices],
            [float(p[0]) for p in predicted_prices],
            [int(d[0]) for d in date_differences],
            [float(PRICE_SCORE_WEIGHT)],
            [float(DATE_SCORE_WEIGHT)],
        ]
    }

    calibration_prices = torch.tensor(
        [[float(rand.randint(200000, 2000000))] for _ in range(batch_size)]
    )
    calibration_predicted = torch.tensor(
        [[p[0] * (1 + (rand.random() - 0.5) * 0.2)] for p in calibration_prices]
    )
    calibration_dates = torch.tensor([[rand.randint(0, 21)] for _ in range(batch_size)])

    calibration_data = {
        "input_data": [
            [float(p[0]) for p in calibration_prices],
            [float(p[0]) for p in calibration_predicted],
            [int(d[0]) for d in calibration_dates],
            [float(PRICE_SCORE_WEIGHT)],
            [float(DATE_SCORE_WEIGHT)],
        ]
    }

    with open("input.json", "w") as f:
        json.dump(input_data, f)
    with open("calibration.json", "w") as f:
        json.dump(calibration_data, f)

    torch.onnx.export(
        circuit,
        (
            dummy_input["actual_price"],
            dummy_input["predicted_price"],
            dummy_input["date_difference"],
            dummy_input["price_score_weight"],
            dummy_input["date_score_weight"],
        ),
        "network.onnx",
        input_names=[
            "actual_price",
            "predicted_price",
            "date_difference",
            "price_score_weight",
            "date_score_weight",
        ],
        output_names=["score"],
        dynamic_axes=None,
    )
