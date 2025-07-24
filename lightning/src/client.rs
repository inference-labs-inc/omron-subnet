use crate::types::{QuicAxonInfo, QuicRequest, QuicResponse, HandshakeRequest, HandshakeResponse, SynapsePacket};
use crate::connection_pool::ConnectionPool;
use pyo3::prelude::*;
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use quinn::{ClientConfig, Endpoint, Connection};
use rustls::{RootCertStore, ClientConfig as RustlsClientConfig};
use std::net::SocketAddr;
use serde_json;
use ed25519_dalek::{Keypair, Signer, SecretKey}; // Keep for commented methods
use base64::{Engine, prelude::BASE64_STANDARD};

pub struct LightningClient {
    wallet_hotkey: String,
    validator_keypair: Option<ed25519_dalek::Keypair>,
    python_signer: Option<PyObject>,
    connection_pool: Arc<RwLock<ConnectionPool>>,
    active_miners: Arc<RwLock<HashMap<String, QuicAxonInfo>>>,
    established_connections: Arc<RwLock<HashMap<String, Connection>>>,
    endpoint: Option<Endpoint>,
}

impl LightningClient {
    pub fn new(wallet_hotkey: String) -> Self {
        Self {
            wallet_hotkey,
            validator_keypair: None,
            python_signer: None,
            connection_pool: Arc::new(RwLock::new(ConnectionPool::new())),
            active_miners: Arc::new(RwLock::new(HashMap::new())),
            established_connections: Arc::new(RwLock::new(HashMap::new())),
            endpoint: None,
        }
    }

    pub fn set_validator_keypair(&mut self, keypair_seed: [u8; 32]) {
        let secret_key = SecretKey::from_bytes(&keypair_seed).expect("Valid seed");
        let public_key = ed25519_dalek::PublicKey::from(&secret_key);
        let keypair = Keypair { secret: secret_key, public: public_key };
        self.validator_keypair = Some(keypair);
        println!("🔑 Validator keypair configured for signing");
    }

    pub fn set_python_signer(&mut self, signer: PyObject) {
        self.python_signer = Some(signer);
        println!("🔑 Python signer configured for bittensor wallet signing");
    }

    async fn call_python_signer(&self, signer: &PyObject, message: &str) -> Result<String, String> {
        Python::with_gil(|py| {
            // Call the Python signing function with the message
            let result = signer.call1(py, (message,))
                .map_err(|e| format!("Python signer call failed: {}", e))?;

            // Extract the signature as bytes
            let signature_bytes: Vec<u8> = result.extract(py)
                .map_err(|e| format!("Failed to extract signature bytes: {}", e))?;

            // Encode as base64 to match expected format
            Ok(BASE64_STANDARD.encode(&signature_bytes))
        })
    }

    pub async fn initialize_connections(&mut self, miners: Vec<QuicAxonInfo>) -> PyResult<()> {
        // Create QUIC client endpoint
        self.create_endpoint().await?;

        let mut active_miners = self.active_miners.write().await;
        let mut pool = self.connection_pool.write().await;

        for miner in miners {
            let miner_key = format!("{}:{}", miner.ip, miner.port);

            match self.establish_connection_with_handshake(&miner).await {
                Ok(connection) => {
                    let connection_id = format!("{}:{}:{}", miner.ip, miner.port,
                        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis());

                    pool.add_connection(&miner_key, connection_id.clone()).await;
                    active_miners.insert(miner_key.clone(), miner);

                    // Store the actual QUIC connection
                    let mut connections = self.established_connections.write().await;
                    connections.insert(miner_key.clone(), connection);

                    println!("✅ Established persistent QUIC connection to miner: {}", miner_key);
                }
                Err(e) => {
                    println!("❌ Failed to connect to miner {}: {}", miner_key, e);
                }
            }
        }

        Ok(())
    }

    pub async fn create_endpoint(&mut self) -> PyResult<()> {
        // Create a simple client config that accepts self-signed certificates
        let _root_store = RootCertStore::empty();

        // Configure to accept any certificate (for self-signed certs)
        let mut tls_config = RustlsClientConfig::builder()
            .with_safe_defaults()
            .with_custom_certificate_verifier(Arc::new(AcceptAnyCertVerifier))
            .with_no_client_auth();

        // Set ALPN protocol to match server - CRITICAL for protocol negotiation
        tls_config.alpn_protocols = vec![b"lightning-quic".to_vec()];

        println!("🔧 Client TLS config created with ALPN: lightning-quic");

        let client_config = ClientConfig::new(Arc::new(tls_config));
        let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap()).unwrap();
        endpoint.set_default_client_config(client_config);
        self.endpoint = Some(endpoint);

        println!("✅ QUIC client endpoint created");
        Ok(())
    }

    async fn establish_connection_with_handshake(&self, miner: &QuicAxonInfo) -> Result<Connection, String> {
        if let Some(endpoint) = &self.endpoint {
            let addr: SocketAddr = format!("{}:{}", miner.ip, miner.port)
                .parse()
                .map_err(|e| format!("Invalid address: {}", e))?;

            println!("🔗 Connecting to miner at {}", addr);

            // Establish QUIC connection
            println!("🔧 Client ALPN protocols configured: lightning-quic");
            let connection = endpoint.connect(addr, "localhost")
                .map_err(|e| format!("Connection failed: {}", e))?
                .await
                .map_err(|e| format!("Connection handshake failed: {}", e))?;

            println!("🤝 QUIC connection established, performing handshake with {}", miner.hotkey);

            // Perform application-level handshake
            let handshake_request = HandshakeRequest {
                validator_hotkey: self.wallet_hotkey.clone(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                signature: self.sign_handshake_message(miner).await?,
            };

            // Send handshake via QUIC stream
            match self.send_handshake(&connection, handshake_request).await {
                Ok(response) => {
                    if response.accepted {
                        println!("✅ Handshake successful with miner {}", miner.hotkey);
                        Ok(connection)
                    } else {
                        Err("Handshake rejected by miner".to_string())
                    }
                }
                Err(e) => Err(format!("Handshake failed: {}", e))
            }
        } else {
            Err("QUIC endpoint not initialized".to_string())
        }
    }

    async fn send_handshake(&self, connection: &Connection, request: HandshakeRequest) -> Result<HandshakeResponse, String> {
        let (mut send, mut recv) = connection.open_bi().await
            .map_err(|e| format!("Failed to open bidirectional stream: {}", e))?;

        // Send handshake request
        let request_json = serde_json::to_string(&request)
            .map_err(|e| format!("Failed to serialize handshake: {}", e))?;

        send.write_all(request_json.as_bytes()).await
            .map_err(|e| format!("Failed to send handshake: {}", e))?;
        send.finish().await
            .map_err(|e| format!("Failed to finish sending handshake: {}", e))?;

        // Receive handshake response
        let buffer = recv.read_to_end(1024 * 1024).await // 1MB limit
            .map_err(|e| format!("Failed to read handshake response: {}", e))?;

        let response_str = String::from_utf8(buffer)
            .map_err(|e| format!("Invalid UTF-8 in response: {}", e))?;

        let response: HandshakeResponse = serde_json::from_str(&response_str)
            .map_err(|e| format!("Failed to parse handshake response: {}", e))?;

        Ok(response)
    }

    async fn sign_handshake_message(&self, _miner: &QuicAxonInfo) -> Result<String, String> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let message = format!("handshake:{}:{}", self.wallet_hotkey, timestamp);

        match &self.validator_keypair {
            Some(keypair) => {
                let signature = keypair.sign(message.as_bytes());
                let signature_bytes = signature.to_bytes();
                Ok(BASE64_STANDARD.encode(&signature_bytes))
            }
            None => {
                // Try to use Python signing callback if available
                if let Some(ref python_signer) = self.python_signer {
                    match self.call_python_signer(python_signer, &message).await {
                        Ok(signature) => {
                            println!("✅ Used bittensor wallet for handshake signature");
                            Ok(signature)
                        }
                        Err(e) => {
                            println!("⚠️ Python signing failed: {}, using dummy signature", e);
                            let dummy_bytes = &message.as_bytes()[..8];
                            Ok(BASE64_STANDARD.encode(dummy_bytes))
                        }
                    }
                } else {
                    println!("⚠️ No validator keypair configured, using dummy signature");
                    // Use base64 format to match server expectations
                    let dummy_bytes = &message.as_bytes()[..8];
                    Ok(BASE64_STANDARD.encode(dummy_bytes))
                }
            }
        }
    }

    pub async fn query_axon(&self, axon_info: QuicAxonInfo, request: QuicRequest) -> PyResult<QuicResponse> {
        let miner_key = format!("{}:{}", axon_info.ip, axon_info.port);

        // Get the established QUIC connection
        let connections = self.established_connections.read().await;
        if let Some(connection) = connections.get(&miner_key) {
            self.send_synapse_packet(connection, &axon_info, request).await
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyConnectionError, _>(
                format!("No persistent QUIC connection to miner: {}", miner_key)
            ))
        }
    }

    async fn send_synapse_packet(
        &self,
        connection: &Connection,
        _axon_info: &QuicAxonInfo,
        request: QuicRequest
    ) -> PyResult<QuicResponse> {
        let (mut send, mut recv) = connection.open_bi().await
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyConnectionError, _>(
                format!("Failed to open stream: {}", e)
            ))?;

        // Create synapse packet
        let synapse_packet = SynapsePacket {
            synapse_type: request.synapse_type.clone(),
            data: request.data.clone(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };

        // Send synapse packet
        let packet_json = serde_json::to_string(&synapse_packet)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to serialize synapse packet: {}", e)
            ))?;

        send.write_all(packet_json.as_bytes()).await
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyConnectionError, _>(
                format!("Failed to send synapse packet: {}", e)
            ))?;
        send.finish().await
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyConnectionError, _>(
                format!("Failed to finish sending: {}", e)
            ))?;

        // Receive response
        let buffer = recv.read_to_end(1024 * 1024).await // 1MB limit
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyConnectionError, _>(
                format!("Failed to read response: {}", e)
            ))?;

        let response_str = String::from_utf8(buffer)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid UTF-8 in response: {}", e)
            ))?;

        // Parse synapse response
        let synapse_response: crate::types::SynapseResponse = serde_json::from_str(&response_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to parse synapse response: {}", e)
            ))?;

        // Convert to QuicResponse format
        let mut response_data = HashMap::new();
        for (key, value) in synapse_response.data {
            response_data.insert(key, value);
        }

        Ok(QuicResponse {
            success: synapse_response.success,
            data: response_data,
            latency_ms: 0.0, // TODO: Calculate actual latency
        })
    }

    pub async fn update_miner_registry(&mut self, miners: Vec<QuicAxonInfo>) -> PyResult<()> {
        let current_miners: HashMap<String, QuicAxonInfo> = miners.iter()
            .map(|m| (format!("{}:{}", m.ip, m.port), m.clone()))
            .collect();

        let mut active_miners = self.active_miners.write().await;
        let mut pool = self.connection_pool.write().await;
        let mut connections = self.established_connections.write().await;

        // Remove deregistered miners
        let active_keys: Vec<String> = active_miners.keys().cloned().collect();
        for key in active_keys {
            if !current_miners.contains_key(&key) {
                println!("🔌 Miner deregistered, closing QUIC connection: {}", key);
                if let Some(connection) = connections.remove(&key) {
                    connection.close(0u32.into(), b"miner_deregistered");
                }
                pool.remove_connection(&key).await;
                active_miners.remove(&key);
            }
        }

        // Add new miners
        for (key, miner) in current_miners {
            if !active_miners.contains_key(&key) {
                println!("🆕 New miner detected, establishing QUIC connection: {}", key);
                match self.establish_connection_with_handshake(&miner).await {
                    Ok(connection) => {
                        let connection_id = format!("{}:{}:{}", miner.ip, miner.port,
                            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis());

                        pool.add_connection(&key, connection_id).await;
                        active_miners.insert(key.clone(), miner);
                        connections.insert(key, connection);
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
        let connections = self.established_connections.read().await;

        let mut stats = HashMap::new();
        stats.insert("total_connections".to_string(), pool.connection_count().await.to_string());
        stats.insert("active_miners".to_string(), active_miners.len().to_string());
        stats.insert("quic_connections".to_string(), connections.len().to_string());

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
        let mut connections = self.established_connections.write().await;

        // Close all QUIC connections
        for (_, connection) in connections.drain() {
            connection.close(0u32.into(), b"client_shutdown");
        }

        pool.close_all().await;
        active_miners.clear();

        println!("🔌 All Lightning QUIC connections closed");
        Ok(())
    }
}

// Custom certificate verifier that accepts any certificate (for self-signed certs)
struct AcceptAnyCertVerifier;

impl rustls::client::ServerCertVerifier for AcceptAnyCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::Certificate,
        _intermediates: &[rustls::Certificate],
        _server_name: &rustls::ServerName,
        _scts: &mut dyn Iterator<Item = &[u8]>,
        _ocsp_response: &[u8],
        _now: std::time::SystemTime,
    ) -> Result<rustls::client::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::ServerCertVerified::assertion())
    }
}
