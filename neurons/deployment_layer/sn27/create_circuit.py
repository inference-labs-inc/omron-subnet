import torch
import json
import random as rand


class Circuit(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.zero = torch.tensor(0.0)
        self.one = torch.tensor(1.0)
        self.two = torch.tensor(2.0)
        self.one_hundred = torch.tensor(100.0)
        self.twenty = torch.tensor(20.0)
        self.nineteen = torch.tensor(19.0)

    def normalize(
        self, val: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor
    ):
        return torch.div(torch.sub(val, min_value), torch.sub(max_value, min_value))

    def percent(self, a: torch.Tensor, b: torch.Tensor):
        return torch.where(
            b == self.zero, self.zero, torch.mul(torch.div(a, b), self.one_hundred)
        )

    def percent_yield(self, a: torch.Tensor, b: torch.Tensor):
        return torch.where(
            a == self.zero,
            self.one_hundred,
            torch.mul(torch.div(torch.sub(b, a), b), self.one_hundred),
        )

    def forward(
        self,
        challenge_attempts: torch.Tensor,
        challenge_successes: torch.Tensor,
        last_20_challenge_failed: torch.Tensor,
        challenge_elapsed_time_avg: torch.Tensor,
        last_20_difficulty_avg: torch.Tensor,
        has_docker: torch.Tensor,
        uid: torch.Tensor,
        allocated_uids: torch.Tensor,
        penalized_uids: torch.Tensor,
        validator_uids: torch.Tensor,
        success_weight: torch.Tensor,
        difficulty_weight: torch.Tensor,
        time_elapsed_weight: torch.Tensor,
        failed_penalty_weight: torch.Tensor,
        allocation_weight: torch.Tensor,
        failed_penalty_exp: torch.Tensor,
        pow_timeout: torch.Tensor,
        pow_min_difficulty: torch.Tensor,
        pow_max_difficulty: torch.Tensor,
    ):

        challenge_attempts = torch.max(challenge_attempts, self.one)
        challenge_successes = torch.max(challenge_successes, self.zero)
        last_20_challenge_failed = torch.max(last_20_challenge_failed, self.zero)
        challenge_elapsed_time_avg = torch.min(challenge_elapsed_time_avg, pow_timeout)
        last_20_difficulty_avg = torch.max(last_20_difficulty_avg, pow_min_difficulty)

        difficulty_val = torch.clamp(
            last_20_difficulty_avg, min=pow_min_difficulty, max=pow_max_difficulty
        )

        difficulty_modifier = self.percent(difficulty_val, pow_max_difficulty)

        difficulty = torch.mul(difficulty_modifier, difficulty_weight)

        successes_ratio = self.percent(challenge_successes, challenge_attempts)
        successes = torch.mul(successes_ratio, success_weight)

        time_elapsed_modifier = self.percent_yield(
            challenge_elapsed_time_avg, pow_timeout
        )
        time_elapsed = torch.mul(time_elapsed_modifier, time_elapsed_weight)

        last_20_challenge_failed_modifier = self.percent(
            last_20_challenge_failed, self.twenty
        )
        failed_penalty = torch.mul(
            torch.mul(
                failed_penalty_weight,
                torch.pow(
                    torch.div(last_20_challenge_failed_modifier, self.one_hundred),
                    failed_penalty_exp,
                ),
            ),
            self.one_hundred,
        )

        allocation_score = torch.mul(difficulty_modifier, allocation_weight)
        allocation_status = torch.zeros_like(uid, dtype=torch.bool)
        for i in range(uid.shape[0]):
            allocation_status[i] = torch.any(torch.eq(uid[i], allocated_uids))

        max_score_challenge = torch.mul(
            self.one_hundred,
            torch.add(
                torch.add(success_weight, difficulty_weight), time_elapsed_weight
            ),
        )
        max_score_allocation = torch.mul(self.one_hundred, allocation_weight)
        max_score = torch.add(max_score_challenge, max_score_allocation)
        final_score = torch.add(
            torch.add(torch.add(difficulty, successes), time_elapsed),
            torch.neg(failed_penalty),
        )

        penalty = torch.logical_not(has_docker)

        final_score = torch.where(
            allocation_status,
            torch.add(
                torch.mul(max_score_challenge, torch.sub(self.one, allocation_weight)),
                allocation_score,
            ),
            torch.where(
                penalty,
                torch.div(
                    torch.add(torch.add(difficulty, successes), time_elapsed),
                    self.two,
                ),
                torch.add(torch.add(difficulty, successes), time_elapsed),
            ),
        )

        return_zero = torch.logical_and(
            torch.logical_or(
                torch.ge(last_20_challenge_failed, self.nineteen),
                torch.eq(challenge_successes, self.zero),
            ),
            torch.logical_not(allocation_status),
        )

        penalty_count = torch.sum(torch.eq(penalized_uids, uid)).to(torch.float32)
        half_validators = torch.div(validator_uids.shape[0], self.two)

        penalty_multiplier = torch.where(
            torch.ge(penalty_count, half_validators),
            self.zero,
            torch.max(
                torch.sub(self.one, torch.div(penalty_count, half_validators)),
                self.zero,
            ),
        )
        return_zero = torch.where(
            torch.ge(penalty_count, half_validators),
            torch.ones_like(return_zero, dtype=torch.bool),
            return_zero,
        )
        final_score = torch.where(
            torch.any(torch.eq(uid, penalized_uids)),
            torch.mul(final_score, penalty_multiplier),
            final_score,
        )

        final_score = torch.max(final_score, self.zero)

        final_score = self.normalize(final_score, self.zero, max_score)
        return torch.where(return_zero, self.zero, final_score)


SUCCESS_WEIGHT = 1
DIFFICULTY_WEIGHT = 1
TIME_ELAPSED_WEIGHT = 0.3
FAILED_PENALTY_WEIGHT = 0.4
ALLOCATION_WEIGHT = 0.21
FAILED_PENALTY_EXP = 1.5

POW_TIMEOUT = 30
POW_MIN_DIFFICULTY = 7
POW_MAX_DIFFICULTY = 30

if __name__ == "__main__":
    circuit = Circuit()
    batch_size = 2

    # Generate random test data tensors
    def generate_tensor(min_val, max_val, batch_size):
        return torch.tensor(
            [[float(rand.randint(min_val, max_val))] for _ in range(batch_size)]
        )

    def generate_random_tensor(min_val, max_val, batch_size):
        return torch.tensor(
            [[min_val + rand.random() * (max_val - min_val)] for _ in range(batch_size)]
        )

    def generate_uid_tensor(batch_size, prob=0.1):
        return torch.tensor(
            [
                [rand.randint(0, 256) if rand.random() < prob else 0]
                for _ in range(batch_size)
            ]
        )

    challenge_attempts = generate_tensor(5, 10, batch_size)
    challenge_successes = generate_tensor(4, 8, batch_size)
    last_20_challenge_failed = generate_tensor(0, 3, batch_size)
    challenge_elapsed_time_avg = generate_random_tensor(4.0, 8.0, batch_size)
    last_20_difficulty_avg = generate_random_tensor(1.5, 2.5, batch_size)
    has_docker = torch.tensor([[True] for _ in range(batch_size)])
    uid = generate_tensor(0, 256, batch_size)
    allocated_uids = generate_uid_tensor(batch_size)
    penalized_uids = generate_uid_tensor(batch_size)
    validator_uids = generate_uid_tensor(batch_size)

    weights = {
        "success_weight": SUCCESS_WEIGHT,
        "difficulty_weight": DIFFICULTY_WEIGHT,
        "time_elapsed_weight": TIME_ELAPSED_WEIGHT,
        "failed_penalty_weight": FAILED_PENALTY_WEIGHT,
        "allocation_weight": ALLOCATION_WEIGHT,
        "failed_penalty_exp": FAILED_PENALTY_EXP,
        "pow_timeout": POW_TIMEOUT,
        "pow_min_difficulty": POW_MIN_DIFFICULTY,
        "pow_max_difficulty": POW_MAX_DIFFICULTY,
    }

    dummy_input = {
        "challenge_attempts": challenge_attempts,
        "challenge_successes": challenge_successes,
        "last_20_challenge_failed": last_20_challenge_failed,
        "challenge_elapsed_time_avg": challenge_elapsed_time_avg,
        "last_20_difficulty_avg": last_20_difficulty_avg,
        "has_docker": has_docker,
        "uid": uid,
        "allocated_uids": allocated_uids,
        "penalized_uids": penalized_uids,
        "validator_uids": validator_uids,
        **{k: torch.tensor([v]) for k, v in weights.items()},
    }

    def prepare_json_data():
        return {
            "input_data": [
                [float(a[0]) for a in challenge_attempts],
                [float(s[0]) for s in challenge_successes],
                [float(f[0]) for f in last_20_challenge_failed],
                [float(t[0]) for t in challenge_elapsed_time_avg],
                [float(d[0]) for d in last_20_difficulty_avg],
                [float(h[0]) for h in has_docker],
                [float(u[0]) for u in uid],
                [[float(x) for x in lst] for lst in allocated_uids.tolist()],
                [[float(x) for x in lst] for lst in penalized_uids.tolist()],
                [[float(x) for x in lst] for lst in validator_uids.tolist()],
                *[[float(v)] for v in weights.values()],
            ]
        }

    input_data = prepare_json_data()
    calibration_data = prepare_json_data()

    # Export data
    for filename, data in [
        ("input.json", input_data),
        ("calibration.json", calibration_data),
    ]:
        with open(filename, "w") as f:
            json.dump(data, f)

    # Export model
    torch.onnx.export(
        circuit,
        dummy_input,
        "network.onnx",
        input_names=[
            "challenge_attempts",
            "challenge_successes",
            "last_20_challenge_failed",
            "challenge_elapsed_time_avg",
            "last_20_difficulty_avg",
            "has_docker",
            "uid",
            "allocated_uids",
            "penalized_uids",
            "validator_uids",
            *weights.keys(),
        ],
        output_names=["score"],
        dynamic_axes=None,
    )

    def run_command(command):
        import subprocess

        try:
            output = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True
            )
            print(f"Running: {command}")
            print(output)
        except subprocess.CalledProcessError as e:
            print(f"Error running {command}:")
            print(e.output)
            exit(1)

    commands = [
        "ezkl gen-settings --input-visibility=public --param-visibility=fixed",
        "ezkl calibrate-settings",
        "ezkl compile-circuit",
        "ezkl setup",
        "ezkl gen-witness",
        "ezkl prove",
        "ezkl verify",
    ]

    for cmd in commands:
        run_command(cmd)
