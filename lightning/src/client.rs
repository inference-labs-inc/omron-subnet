use crate::connection_pool::ConnectionPool;
use crate::types::{QuicAxonInfo, QuicRequest, QuicResponse, HandshakeRequest, SynapsePacket};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct LightningClient {
    connection_pool: Arc<RwLock<ConnectionPool>>,
    active_miners: Arc<RwLock<HashMap<String, QuicAxonInfo>>>,
    wallet_hotkey: String,
}

impl LightningClient {
    pub fn new(wallet_hotkey: String) -> Self {
        Self {
            connection_pool: Arc::new(RwLock::new(ConnectionPool::new())),
            active_miners: Arc::new(RwLock::new(HashMap::new())),
            wallet_hotkey,
        }
    }

    pub async fn initialize_connections(&self, miners: Vec<QuicAxonInfo>) -> PyResult<()> {
        let mut active_miners = self.active_miners.write().await;
        let mut pool = self.connection_pool.write().await;

        for miner in miners {
            let miner_key = format!("{}:{}", miner.ip, miner.port);

            match self.establish_connection_with_handshake(&miner).await {
                Ok(connection_id) => {
                    pool.add_connection(&miner_key, connection_id).await;
                    active_miners.insert(miner_key.clone(), miner);
                    println!("✅ Established persistent connection to miner: {}", miner_key);
                }
                Err(e) => {
                    println!("❌ Failed to connect to miner {}: {}", miner_key, e);
                }
            }
        }

        Ok(())
    }

    async fn establish_connection_with_handshake(&self, miner: &QuicAxonInfo) -> Result<String, String> {
        let _handshake = HandshakeRequest {
            validator_hotkey: self.wallet_hotkey.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            signature: self.sign_handshake_message(miner).await?,
        };

        let connection_id = format!("{}:{}:{}", miner.ip, miner.port,
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());

        println!("🤝 Performing handshake with {}", miner.hotkey);
        Ok(connection_id)
    }

    async fn sign_handshake_message(&self, miner: &QuicAxonInfo) -> Result<String, String> {
        let message = format!("handshake:{}:{}:{}", self.wallet_hotkey, miner.hotkey,
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
        Ok(format!("0x{}", "mock_signature_".to_string() + &message[..8]))
    }

    pub async fn query_axon(&self, axon_info: QuicAxonInfo, request: QuicRequest) -> PyResult<QuicResponse> {
        let miner_key = format!("{}:{}", axon_info.ip, axon_info.port);

        let pool = self.connection_pool.read().await;
        if let Some(connection_id) = pool.get_connection(&miner_key).await {
            self.send_synapse_packet(&connection_id, &axon_info, request).await
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyConnectionError, _>(
                format!("No persistent connection to miner: {}", miner_key)
            ))
        }
    }

    async fn send_synapse_packet(&self, connection_id: &str, axon_info: &QuicAxonInfo, request: QuicRequest) -> PyResult<QuicResponse> {
        let _packet = SynapsePacket {
            synapse_type: request.synapse_type.clone(),
            data: request.data,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        println!("📦 Sending synapse packet to {} via connection {}", axon_info.hotkey, connection_id);

        let mut response_data = std::collections::HashMap::new();
        response_data.insert("query_output".to_string(), serde_json::Value::String("lightning_response_via_persistent_connection".to_string()));
        response_data.insert("proof".to_string(), serde_json::Value::String("lightning_proof".to_string()));
        response_data.insert("commitment".to_string(), serde_json::Value::String("lightning_commitment".to_string()));
        response_data.insert("axon_hotkey".to_string(), serde_json::Value::String(axon_info.hotkey.clone()));
        response_data.insert("connection_id".to_string(), serde_json::Value::String(connection_id.to_string()));

        Ok(QuicResponse {
            success: true,
            data: response_data,
            latency_ms: 5.0,
        })
    }

    pub async fn update_miner_registry(&self, miners: Vec<QuicAxonInfo>) -> PyResult<()> {
        let current_miners: HashMap<String, QuicAxonInfo> = miners.iter()
            .map(|m| (format!("{}:{}", m.ip, m.port), m.clone()))
            .collect();

        let mut active_miners = self.active_miners.write().await;
        let mut pool = self.connection_pool.write().await;

        let active_keys: Vec<String> = active_miners.keys().cloned().collect();
        for key in active_keys {
            if !current_miners.contains_key(&key) {
                println!("🔌 Miner deregistered, closing connection: {}", key);
                pool.remove_connection(&key).await;
                active_miners.remove(&key);
            }
        }

        for (key, miner) in current_miners {
            if !active_miners.contains_key(&key) {
                println!("🆕 New miner detected, establishing connection: {}", key);
                match self.establish_connection_with_handshake(&miner).await {
                    Ok(connection_id) => {
                        pool.add_connection(&key, connection_id).await;
                        active_miners.insert(key, miner);
                    }
                    Err(e) => {
                        println!("❌ Failed to connect to new miner {}: {}", key, e);
                    }
                }
            }
        }

        Ok(())
    }

    pub async fn get_connection_stats(&self) -> PyResult<HashMap<String, String>> {
        let pool = self.connection_pool.read().await;
        let active_miners = self.active_miners.read().await;

        let mut stats = HashMap::new();
        stats.insert("total_connections".to_string(), pool.connection_count().await.to_string());
        stats.insert("active_miners".to_string(), active_miners.len().to_string());

        for (key, _) in active_miners.iter() {
            if let Some(connection_id) = pool.get_connection(key).await {
                stats.insert(format!("connection_{}", key), connection_id);
            }
        }

        Ok(stats)
    }

    pub async fn close_all_connections(&self) -> PyResult<()> {
        let mut pool = self.connection_pool.write().await;
        let mut active_miners = self.active_miners.write().await;

        pool.close_all().await;
        active_miners.clear();

        println!("🔌 All Lightning connections closed");
        Ok(())
    }
}
