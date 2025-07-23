use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct SerializedRequest {
    pub synapse_data: HashMap<String, serde_json::Value>,
    pub headers: HashMap<String, String>,
    pub endpoint: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SerializedResponse {
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body: serde_json::Value,
}

pub fn serialize_request_msgpack(request: &SerializedRequest) -> Result<Vec<u8>> {
    Ok(rmp_serde::to_vec(request)?)
}

pub fn deserialize_response_msgpack(data: &[u8]) -> Result<SerializedResponse> {
    Ok(rmp_serde::from_slice(data)?)
}

pub fn serialize_request_json(request: &SerializedRequest) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(request)?)
}

pub fn deserialize_response_json(data: &[u8]) -> Result<SerializedResponse> {
    Ok(serde_json::from_slice(data)?)
}

// Helper function to determine if we should use msgpack or JSON
pub fn should_use_msgpack(headers: &HashMap<String, String>) -> bool {
    headers.get("content-type")
        .map(|ct| ct.contains("application/msgpack"))
        .unwrap_or(false)
}
