#![cfg_attr(feature = "guest", no_std)]
#![no_main]
use libm::tanf;

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

pub const BATCH_SIZE: usize = 256;

const RATE_OF_DECAY: f32 = 0.4;
const RATE_OF_RECOVERY: f32 = 0.1;
const RESPONSE_TIME_WEIGHT: f32 = 1.0;
const PROOF_SIZE_WEIGHT: f32 = 0.0;
const PROOF_SIZE_THRESHOLD: f32 = 3648.0;
const FLATTENING_COEFFICIENT: f32 = 0.9;
const MAXIMUM_RESPONSE_TIME_DECIMAL: f32 = 0.99;

#[derive(Clone, Serialize, Deserialize)]
pub struct ScoreInput {
    #[serde(with = "BigArray")]
    pub maximum_score: [f32; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub previous_score: [f32; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub verified: [bool; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub proof_size: [f32; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub response_time: [f32; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub maximum_response_time: [f32; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub minimum_response_time: [f32; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub validator_uid: [i16; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub block_number: [i32; BATCH_SIZE],
    #[serde(with = "BigArray")]
    pub miner_uid: [i16; BATCH_SIZE],
    pub uid_responsible_for_proof: i16,
}

#[derive(Serialize, Deserialize)]
pub struct ScoreOutput(#[serde(with = "BigArray")] pub [f32; BATCH_SIZE]);

#[jolt::provable(
    stack_size = 1_000_000,
    memory_size = 1_000_000,
    max_input_size = 16384
)]
fn score(input: ScoreInput) -> ScoreOutput {
    let batch_size = input.maximum_score.len();
    let mut new_scores = [0.0; BATCH_SIZE];

    for i in 0..batch_size {
        new_scores[i] = calculate_single_score(
            input.maximum_score[i],
            input.previous_score[i],
            input.verified[i],
            input.proof_size[i],
            input.response_time[i],
            input.maximum_response_time[i],
            input.minimum_response_time[i],
            input.validator_uid[i],
            input.block_number[i],
            input.miner_uid[i],
        );
    }

    ScoreOutput(new_scores)
}

fn calculate_single_score(
    maximum_score: f32,
    previous_score: f32,
    verified: bool,
    proof_size: f32,
    response_time: f32,
    maximum_response_time: f32,
    minimum_response_time: f32,
    _validator_uid: i16,
    _block_number: i32,
    _miner_uid: i16,
) -> f32 {
    if !verified {
        return previous_score * RATE_OF_DECAY;
    }

    let response_time_normalized: f32 = clamp(
        (response_time - minimum_response_time) / (maximum_response_time - minimum_response_time),
        0.0,
        MAXIMUM_RESPONSE_TIME_DECIMAL,
    );

    let response_time_reward_metric =
        RESPONSE_TIME_WEIGHT * (normalized_tangent_curve(response_time_normalized));

    let proof_size_reward_metric =
        PROOF_SIZE_WEIGHT * clamp(proof_size / PROOF_SIZE_THRESHOLD, 0.0, 1.0);

    let calculated_score_fraction = clamp(
        response_time_reward_metric - proof_size_reward_metric,
        0.0,
        1.0,
    );

    let new_maximum_score = maximum_score * calculated_score_fraction;
    let distance_from_score = new_maximum_score - previous_score;
    let change_in_score = RATE_OF_RECOVERY * distance_from_score;
    let new_score = previous_score + change_in_score;

    return new_score;
}

fn shifted_tan(x: f32) -> f32 {
    tanf((x - 0.5) * 3.141592653589793238462643383279502884 * FLATTENING_COEFFICIENT)
}

fn tan_shift_difference(x: f32) -> f32 {
    shifted_tan(x) - shifted_tan(0.0)
}

fn normalized_tangent_curve(x: f32) -> f32 {
    tan_shift_difference(x) / tan_shift_difference(1.0)
}

fn clamp(input: f32, lower: f32, upper: f32) -> f32 {
    if input < lower {
        return lower;
    }
    if input > upper {
        return upper;
    }
    return input;
}
