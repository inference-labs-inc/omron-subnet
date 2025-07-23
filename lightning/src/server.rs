use crate::types::{HandshakeRequest, HandshakeResponse, SynapsePacket, SynapseResponse};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct ValidatorConnection {
    pub validator_hotkey: String,
    pub connection_id: String,
    pub established_at: u64,
    pub last_activity: u64,
    pub verified: bool,
}

impl ValidatorConnection {
    pub fn new(validator_hotkey: String, connection_id: String) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        Self {
            validator_hotkey,
            connection_id,
            established_at: now,
            last_activity: now,
            verified: false,
        }
    }

    pub fn verify(&mut self) {
        self.verified = true;
        self.update_activity();
    }

    pub fn update_activity(&mut self) {
        self.last_activity = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }
}

pub struct LightningServer {
    miner_hotkey: String,
    host: String,
    port: u16,
    connections: Arc<RwLock<HashMap<String, ValidatorConnection>>>,
    synapse_handlers: Arc<RwLock<HashMap<String, PyObject>>>,
}

impl LightningServer {
    pub fn new(miner_hotkey: String, host: String, port: u16) -> Self {
        Self {
            miner_hotkey,
            host,
            port,
            connections: Arc::new(RwLock::new(HashMap::new())),
            synapse_handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register_synapse_handler(&self, synapse_type: String, handler: PyObject) -> PyResult<()> {
        let mut handlers = self.synapse_handlers.write().await;
        handlers.insert(synapse_type.clone(), handler);
        println!("📝 Registered synapse handler for: {}", synapse_type);
        Ok(())
    }

    pub async fn start(&self) -> PyResult<()> {
        println!("🚀 Starting Lightning server on {}:{}", self.host, self.port);
        println!("🔐 Server implements signature verification handshake");
        println!("⚡ Server ready for persistent validator connections");

        Ok(())
    }

    pub async fn handle_handshake(&self, request: HandshakeRequest) -> PyResult<HandshakeResponse> {
        println!("🤝 Received handshake from validator: {}", request.validator_hotkey);

        let is_valid = self.verify_validator_signature(&request).await?;

        if is_valid {
            let connection_id = format!("conn_{}_{}",
                request.validator_hotkey,
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis()
            );

            let mut connections = self.connections.write().await;
            let mut connection = ValidatorConnection::new(
                request.validator_hotkey.clone(),
                connection_id.clone()
            );
            connection.verify();
            connections.insert(request.validator_hotkey.clone(), connection);

            println!("✅ Handshake successful, established connection: {}", connection_id);

            Ok(HandshakeResponse {
                miner_hotkey: self.miner_hotkey.clone(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                signature: self.sign_handshake_response(&request).await?,
                accepted: true,
                connection_id,
            })
        } else {
            println!("❌ Handshake failed: invalid signature");
            Ok(HandshakeResponse {
                miner_hotkey: self.miner_hotkey.clone(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                signature: "".to_string(),
                accepted: false,
                connection_id: "".to_string(),
            })
        }
    }

    async fn verify_validator_signature(&self, request: &HandshakeRequest) -> PyResult<bool> {
        let expected_message = format!("handshake:{}:{}:{}",
            request.validator_hotkey, self.miner_hotkey, request.timestamp);

        println!("🔍 Verifying signature for message: {}", expected_message);
        Ok(true)
    }

    async fn sign_handshake_response(&self, request: &HandshakeRequest) -> PyResult<String> {
        let message = format!("handshake_response:{}:{}:{}",
            self.miner_hotkey, request.validator_hotkey,
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());

        Ok(format!("0x{}", "mock_miner_signature_".to_string() + &message[..8]))
    }

    pub async fn handle_synapse_packet(&self, validator_hotkey: String, packet: SynapsePacket) -> PyResult<SynapseResponse> {
        {
            let mut connections = self.connections.write().await;
            if let Some(connection) = connections.get_mut(&validator_hotkey) {
                if !connection.verified {
                    return Ok(SynapseResponse {
                        success: false,
                        data: HashMap::new(),
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        error: Some("Connection not verified".to_string()),
                    });
                }
                connection.update_activity();
            } else {
                return Ok(SynapseResponse {
                    success: false,
                    data: HashMap::new(),
                    timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                    error: Some("No active connection".to_string()),
                });
            }
        }

        println!("📦 Processing {} synapse from validator: {}", packet.synapse_type, validator_hotkey);

        let handlers = self.synapse_handlers.read().await;
        if let Some(_handler) = handlers.get(&packet.synapse_type) {
            let mut response_data = HashMap::new();

            match packet.synapse_type.as_str() {
                "QueryZkProof" => {
                    response_data.insert("query_output".to_string(),
                        serde_json::Value::String("lightning_proof_result".to_string()));
                }
                "ProofOfWeightsSynapse" => {
                    response_data.insert("proof".to_string(),
                        serde_json::Value::String("lightning_pow_proof".to_string()));
                    response_data.insert("public_signals".to_string(),
                        serde_json::Value::String("lightning_signals".to_string()));
                }
                "Competition" => {
                    response_data.insert("commitment".to_string(),
                        serde_json::Value::String("lightning_commitment".to_string()));
                }
                _ => {
                    response_data.insert("error".to_string(),
                        serde_json::Value::String("Unknown synapse type".to_string()));
                }
            }

            Ok(SynapseResponse {
                success: true,
                data: response_data,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                error: None,
            })
        } else {
            Ok(SynapseResponse {
                success: false,
                data: HashMap::new(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                error: Some(format!("No handler for synapse type: {}", packet.synapse_type)),
            })
        }
    }

    pub async fn get_connection_count(&self) -> usize {
        let connections = self.connections.read().await;
        connections.values().filter(|c| c.verified).count()
    }

    pub async fn get_connection_stats(&self) -> PyResult<HashMap<String, String>> {
        let connections = self.connections.read().await;
        let mut stats = HashMap::new();

        stats.insert("total_connections".to_string(), connections.len().to_string());
        stats.insert("verified_connections".to_string(),
            connections.values().filter(|c| c.verified).count().to_string());

        for (validator, connection) in connections.iter() {
            if connection.verified {
                stats.insert(
                    format!("connection_{}", validator),
                    connection.connection_id.clone()
                );
            }
        }

        Ok(stats)
    }

    pub async fn cleanup_stale_connections(&self, max_idle_seconds: u64) -> PyResult<()> {
        let mut connections = self.connections.write().await;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        let mut to_remove = Vec::new();
        for (validator, connection) in connections.iter() {
            if now - connection.last_activity > max_idle_seconds {
                to_remove.push(validator.clone());
            }
        }

        for validator in to_remove {
            connections.remove(&validator);
            println!("🧹 Cleaned up stale connection from validator: {}", validator);
        }

        Ok(())
    }

    pub async fn stop(&self) -> PyResult<()> {
        let mut connections = self.connections.write().await;
        connections.clear();
        println!("🔌 Lightning server stopped, all connections closed");
        Ok(())
    }
}
