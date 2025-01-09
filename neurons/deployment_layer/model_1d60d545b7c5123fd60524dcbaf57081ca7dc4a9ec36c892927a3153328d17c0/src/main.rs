use clap::{Parser, Subcommand};
use guest::{build_score_sn27_v005, ScoreInput, ScoreOutput, BATCH_SIZE};
use jolt_core::jolt::vm::rv32i_vm::{JoltHyperKZGProof, Serializable};
use log::{debug, error, info, trace};
use postcard;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use serde_json;
use std::{self, process::ExitCode};
#[derive(Clone, Serialize, Deserialize)]
pub struct DecodedInputs {
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
#[derive(Serialize, Deserialize, Debug)]
struct DecodedOutputs {
    #[serde(with = "BigArray")]
    score: [f32; BATCH_SIZE],
}

#[derive(Parser, Clone)]
struct Args {
    #[arg(short, long, default_value = "input.json")]
    input: String,
    #[arg(short, long, default_value = "output.json")]
    output: String,
    #[arg(short, long, default_value = "proof.bin")]
    proof: String,
}

#[derive(Subcommand, Clone)]
enum Commands {
    Prove(Args),
    Verify(Args),
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}
fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    let (prove_score, verify_score) = build_score_sn27_v005();

    match &cli.command {
        Commands::Prove(args) => match prove_command(args, prove_score) {
            Ok(_) => {
                info!("Proof generated successfully");
                ExitCode::SUCCESS
            }
            Err(e) => {
                error!("Error in prove command: {}", e);
                ExitCode::FAILURE
            }
        },
        Commands::Verify(args) => match verify_command(args, verify_score) {
            Ok(is_valid) => {
                if is_valid {
                    info!("Proof is valid");
                    ExitCode::SUCCESS
                } else {
                    error!("Proof is invalid");
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                error!("Error in verify command: {}", e);
                error!("Proof is invalid");
                ExitCode::FAILURE
            }
        },
    }
}

fn prove_command(
    args: &Args,
    prove_score: impl Fn(ScoreInput) -> (ScoreOutput, JoltHyperKZGProof),
) -> Result<(), Box<dyn std::error::Error>> {
    let input = read_input_from_file(&args.input)?;

    let (output, proof) = prove_score(input);

    proof.save_to_file(&args.proof)?;

    let output_str = serde_json::to_string(&output)?;
    std::fs::write(&args.output, output_str)?;

    debug!("Proof inputs: {:?}", proof.proof.program_io.inputs);
    debug!("Proof outputs: {:?}", proof.proof.program_io.outputs);
    let (decoded_inputs, decoded_outputs) = decode_proof_io(&proof);
    trace!(
        "challenge_attempts: {:?}",
        decoded_inputs.challenge_attempts
    );
    trace!(
        "challenge_successes: {:?}",
        decoded_inputs.challenge_successes
    );
    trace!(
        "last_20_challenge_failed: {:?}",
        decoded_inputs.last_20_challenge_failed
    );
    trace!(
        "challenge_elapsed_time_avg: {:?}",
        decoded_inputs.challenge_elapsed_time_avg
    );
    trace!(
        "challenge_difficulty_avg: {:?}",
        decoded_inputs.challenge_difficulty_avg
    );
    trace!("has_docker: {:?}", decoded_inputs.has_docker);
    trace!("allocated_hotkey: {:?}", decoded_inputs.allocated_hotkey);
    trace!(
        "penalized_hotkey_count: {:?}",
        decoded_inputs.penalized_hotkey_count
    );
    trace!("half_validators: {:?}", decoded_inputs.half_validators);
    trace!("nonce: {:?}", decoded_inputs.nonce);

    trace!("new_score: {:?}", decoded_outputs.score);

    debug!("Proof generated and saved successfully");
    Ok(())
}

fn verify_command(
    args: &Args,
    verify_score: impl Fn(JoltHyperKZGProof) -> bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    let input = read_input_from_file(&args.input)?;
    let output = read_output_from_file(&args.output)?;

    trace!("Output: {:?}", output.0);

    let mut proof = JoltHyperKZGProof::from_file(&args.proof)?;

    // Replace proof io with the input and output passed in
    // to verify that these inputs and outputs were the ones used to generate the proof
    let (decoded_inputs, decoded_outputs) = decode_proof_io(&proof);
    proof.proof.program_io.inputs = postcard::to_stdvec(&input).unwrap();
    proof.proof.program_io.outputs = postcard::to_stdvec(&output).unwrap();

    debug!("Decoded proof inputs:");
    trace!(
        "challenge_attempts: {:?}",
        decoded_inputs.challenge_attempts
    );
    trace!(
        "challenge_successes: {:?}",
        decoded_inputs.challenge_successes
    );
    trace!(
        "last_20_challenge_failed: {:?}",
        decoded_inputs.last_20_challenge_failed
    );
    trace!(
        "challenge_elapsed_time_avg: {:?}",
        decoded_inputs.challenge_elapsed_time_avg
    );
    trace!(
        "challenge_difficulty_avg: {:?}",
        decoded_inputs.challenge_difficulty_avg
    );
    trace!("has_docker: {:?}", decoded_inputs.has_docker);
    trace!("allocated_hotkey: {:?}", decoded_inputs.allocated_hotkey);
    trace!(
        "penalized_hotkey_count: {:?}",
        decoded_inputs.penalized_hotkey_count
    );
    trace!("half_validators: {:?}", decoded_inputs.half_validators);
    trace!("nonce: {:?}", decoded_inputs.nonce);
    trace!("Decoded proof outputs: {:?}", decoded_outputs);

    let is_valid = verify_score(proof);

    debug!("Proof is {}", if is_valid { "valid" } else { "invalid" });
    Ok(is_valid)
}

fn read_input_from_file(path: &str) -> Result<ScoreInput, Box<dyn std::error::Error>> {
    let file_content = std::fs::read_to_string(path)?;
    serde_json::from_str(&file_content).map_err(Into::into)
}

fn read_output_from_file(path: &str) -> Result<ScoreOutput, Box<dyn std::error::Error>> {
    let file_content = std::fs::read_to_string(path)?;
    serde_json::from_str(&file_content).map_err(Into::into)
}

fn decode_proof_io(proof: &JoltHyperKZGProof) -> (DecodedInputs, DecodedOutputs) {
    let decoded_inputs: DecodedInputs =
        postcard::from_bytes(&proof.proof.program_io.inputs).expect("Failed to decode inputs");
    let decoded_outputs: DecodedOutputs =
        postcard::from_bytes(&proof.proof.program_io.outputs).expect("Failed to decode outputs");
    (decoded_inputs, decoded_outputs)
}
