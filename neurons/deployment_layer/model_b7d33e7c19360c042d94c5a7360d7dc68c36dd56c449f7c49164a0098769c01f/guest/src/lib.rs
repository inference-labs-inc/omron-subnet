#![cfg_attr(feature = "guest", no_std)]
#![no_main]
use libm::{fmaxf, fminf, powf};

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

pub const BATCH_SIZE: usize = 256;

#[derive(Clone, Serialize, Deserialize)]
pub struct ScoreInput {
    pub success_weight: f32,
    pub difficulty_weight: f32,
    pub time_elapsed_weight: f32,
    pub failed_penalty_weight: f32,
    pub allocation_weight: f32,
    pub pow_min_difficulty: u16,
    pub pow_max_difficulty: u16,
    pub pow_timeout: f32,
    pub max_score_challenge: f32,
    pub max_score: f32,
    pub failed_penalty_exp: f32,
    #[serde(with = "BigArray")]
    pub challenge_attempts: [u16; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub challenge_successes: [u16; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub last_20_challenge_failed: [u16; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub challenge_elapsed_time_avg: [f32; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub challenge_difficulty_avg: [f32; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub has_docker: [bool; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub allocated_hotkey: [bool; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub penalized_hotkey_count: [u16; BATCH_SIZE],
    pub half_validators: f32,
    pub nonce: u128,
}

struct ScoreParams {
    success_weight: f32,
    difficulty_weight: f32,
    time_elapsed_weight: f32,
    failed_penalty_weight: f32,
    allocation_weight: f32,
    pow_min_difficulty: u16,
    pow_max_difficulty: u16,
    pow_timeout: f32,
    max_score_challenge: f32,
    max_score: f32,
    failed_penalty_exp: f32,
    challenge_attempts: u16,
    challenge_successes: u16,
    challenge_elapsed_time_avg: f32,
    challenge_difficulty_avg: f32,
    last_20_challenge_failed: u16,
    has_docker: bool,
    allocated_hotkey: bool,
    penalized_hotkey_count: u16,
    half_validators: f32,
}

#[derive(Serialize, Deserialize)]
pub struct ScoreOutput(#[serde(with = "BigArray")] pub [f32; BATCH_SIZE]);

#[jolt::provable(
    stack_size = 100000,
    memory_size = 100000,
    max_input_size = 16384,
    max_output_size = 16384
)]
fn score_sn27(input: ScoreInput) -> ScoreOutput {
    let batch_size = BATCH_SIZE;
    let mut new_scores = [0.0; BATCH_SIZE];
    for i in 0..batch_size {
        new_scores[i] = calculate_single_score(ScoreParams {
            success_weight: input.success_weight,
            difficulty_weight: input.difficulty_weight,
            time_elapsed_weight: input.time_elapsed_weight,
            failed_penalty_weight: input.failed_penalty_weight,
            allocation_weight: input.allocation_weight,
            pow_min_difficulty: input.pow_min_difficulty,
            pow_max_difficulty: input.pow_max_difficulty,
            pow_timeout: input.pow_timeout,
            max_score_challenge: input.max_score_challenge,
            max_score: input.max_score,
            failed_penalty_exp: input.failed_penalty_exp,
            challenge_attempts: input.challenge_attempts[i],
            challenge_successes: input.challenge_successes[i],
            challenge_elapsed_time_avg: input.challenge_elapsed_time_avg[i],
            challenge_difficulty_avg: input.challenge_difficulty_avg[i],
            last_20_challenge_failed: input.last_20_challenge_failed[i],
            has_docker: input.has_docker[i],
            allocated_hotkey: input.allocated_hotkey[i],
            penalized_hotkey_count: input.penalized_hotkey_count[i],
            half_validators: input.half_validators,
        });
    }

    ScoreOutput(new_scores)
}

fn calculate_single_score(params: ScoreParams) -> f32 {
    if (params.last_20_challenge_failed >= 19 || params.challenge_successes == 0)
        && !params.allocated_hotkey
    {
        return 0.0;
    }

    let difficulty_val = fmaxf(
        fminf(
            params.challenge_difficulty_avg,
            params.pow_max_difficulty as f32,
        ),
        params.pow_min_difficulty as f32,
    );
    let difficulty_modifier = percent(difficulty_val, params.pow_max_difficulty as f32);

    let difficulty = difficulty_modifier * params.difficulty_weight;

    let successes_ratio = percent(
        params.challenge_successes as f32,
        params.challenge_attempts as f32,
    );
    let successes = successes_ratio * params.success_weight;

    let time_elapsed_modifier =
        percent_yield(params.challenge_elapsed_time_avg, params.pow_timeout);
    let time_elapsed = time_elapsed_modifier * params.time_elapsed_weight;

    let last_20_challenge_failed_modifier = percent(params.last_20_challenge_failed as f32, 20.0);

    let failed_penalty = params.failed_penalty_weight
        * powf(
            last_20_challenge_failed_modifier / 100.0,
            params.failed_penalty_exp,
        )
        * 100.0;

    let allocation_score = difficulty_modifier * params.allocation_weight;

    let mut final_score = if params.allocated_hotkey {
        params.max_score_challenge * (1.0 - params.allocation_weight) + allocation_score
    } else {
        let intermediate_score = difficulty + successes + time_elapsed - failed_penalty;
        if !params.has_docker {
            intermediate_score / 2.0
        } else {
            intermediate_score
        }
    };

    if params.penalized_hotkey_count > 0 {
        if params.penalized_hotkey_count as f32 >= params.half_validators {
            final_score = 0.0;
        } else {
            final_score *= fmaxf(
                1.0 - (params.penalized_hotkey_count as f32 / params.half_validators),
                0.0,
            );
        }
    }

    final_score = fmaxf(0.0, final_score);

    normalize(final_score, 0.0, params.max_score)
}

fn normalize(val: f32, min_value: f32, max_value: f32) -> f32 {
    (val - min_value) / (max_value - min_value)
}

fn percent(val: f32, max: f32) -> f32 {
    if max == 0.0 {
        return 0.0;
    }
    100.0 * (val / max)
}

fn percent_yield(val: f32, max: f32) -> f32 {
    if val == 0.0 {
        return 100.0;
    }
    100.0 * ((max - val) / max)
}
