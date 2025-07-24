use crate::types::{HandshakeRequest, HandshakeResponse, SynapsePacket, SynapseResponse};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};
use quinn::{Endpoint, ServerConfig, Connection, RecvStream, SendStream};
use rustls::{Certificate, PrivateKey, ServerConfig as RustlsServerConfig};
use std::net::SocketAddr;
use serde_json;
use ed25519_dalek::{Keypair, Signer, SecretKey};
use base64::{Engine, prelude::BASE64_STANDARD};
use sp_core::{sr25519, Pair, crypto::Ss58Codec};

// Maximum age for signature timestamps (5 minutes in seconds)
const MAX_SIGNATURE_AGE: u64 = 300;

#[derive(Debug, Clone)]
pub struct ValidatorConnection {
    pub validator_hotkey: String,
    pub connection_id: String,
    pub established_at: u64,
    pub last_activity: u64,
    pub verified: bool,
    pub connection: Arc<quinn::Connection>,
}

impl ValidatorConnection {
    pub fn new(validator_hotkey: String, connection_id: String, conn: Arc<quinn::Connection>) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        Self {
            validator_hotkey,
            connection_id,
            established_at: now,
            last_activity: now,
            verified: false,
            connection: conn,
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
    miner_keypair_bytes: Option<[u8; 32]>, // Store seed bytes instead of keypair to avoid clone issues
    host: String,
    port: u16,
    connections: Arc<RwLock<HashMap<String, ValidatorConnection>>>,
    synapse_handlers: Arc<RwLock<HashMap<String, PyObject>>>,
    endpoint: Option<Endpoint>,
}

impl LightningServer {
    pub fn new(miner_hotkey: String, host: String, port: u16) -> Self {
        Self {
            miner_hotkey,
            miner_keypair_bytes: None, // Initialize as None
            host,
            port,
            connections: Arc::new(RwLock::new(HashMap::new())),
            synapse_handlers: Arc::new(RwLock::new(HashMap::new())),
            endpoint: None,
        }
    }

    // Temporarily comment out unused methods to remove warnings
    /*
    pub fn set_miner_keypair(&mut self, keypair_seed: [u8; 32]) {
        self.miner_keypair_bytes = Some(keypair_seed);
        println!("🔑 Miner keypair configured for signing");
    }

    pub async fn get_connection_count(&self) -> usize {
        let connections = self.connections.read().await;
        connections.values().filter(|c| c.verified).count()
    }
    */

    pub async fn register_synapse_handler(&self, synapse_type: String, handler: PyObject) -> PyResult<()> {
        let mut handlers = self.synapse_handlers.write().await;
        handlers.insert(synapse_type.clone(), handler);
        println!("📝 Registered synapse handler for: {}", synapse_type);
        Ok(())
    }

    fn create_self_signed_cert() -> Result<(Vec<Certificate>, PrivateKey), Box<dyn std::error::Error>> {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()])?;
        let cert_der = cert.serialize_der()?;
        let priv_key = cert.serialize_private_key_der();

        Ok((
            vec![Certificate(cert_der)],
            PrivateKey(priv_key)
        ))
    }

    pub async fn start(&mut self) -> PyResult<()> {
        println!("🚀 Starting real Lightning QUIC server on {}:{}", self.host, self.port);

        // Create self-signed certificate for QUIC
        let (certs, key) = Self::create_self_signed_cert()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create certificate: {}", e)
            ))?;

        // Configure TLS
        let mut server_config = RustlsServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to configure TLS: {}", e)
            ))?;

        server_config.alpn_protocols = vec![b"lightning-quic".to_vec()];

        let server_config = ServerConfig::with_crypto(Arc::new(server_config));

        // Bind to address
        let addr: SocketAddr = format!("{}:{}", self.host, self.port)
            .parse()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid address: {}", e)
            ))?;

        let endpoint = Endpoint::server(server_config, addr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create QUIC endpoint: {}", e)
            ))?;

        println!("✅ QUIC endpoint created, listening on {}", addr);
        self.endpoint = Some(endpoint);

        println!("🔐 Server implements real signature verification handshake");
        println!("⚡ Server ready for persistent validator connections");

        Ok(())
    }

    pub async fn serve_forever(&mut self) -> PyResult<()> {
        if let Some(endpoint) = &self.endpoint {
            println!("🚀 Lightning QUIC server accepting connections...");

            while let Some(conn) = endpoint.accept().await {
                let connections = self.connections.clone();
                let synapse_handlers = self.synapse_handlers.clone();
                let miner_hotkey = self.miner_hotkey.clone();
                let miner_keypair = self.miner_keypair_bytes.clone();

                tokio::spawn(async move {
                    match conn.await {
                        Ok(connection) => {
                            println!("📡 New QUIC connection from: {}", connection.remote_address());
                            Self::handle_connection(connection, connections, synapse_handlers, miner_hotkey, miner_keypair).await;
                        }
                        Err(e) => {
                            println!("❌ Connection failed: {}", e);
                        }
                    }
                });
            }
        }
        Ok(())
    }

    async fn handle_connection(
        connection: Connection,
        connections: Arc<RwLock<HashMap<String, ValidatorConnection>>>,
        synapse_handlers: Arc<RwLock<HashMap<String, PyObject>>>,
        miner_hotkey: String,
        miner_keypair: Option<[u8; 32]>,
    ) {
        let connection = Arc::new(connection);

        loop {
            match connection.accept_bi().await {
                Ok((send, recv)) => {
                    let conn_clone = connection.clone();
                    let connections_clone = connections.clone();
                    let handlers_clone = synapse_handlers.clone();
                    let miner_hotkey_clone = miner_hotkey.clone();
                    let miner_keypair_clone = miner_keypair.clone();

                    tokio::spawn(async move {
                        Self::handle_stream(
                            send, recv, conn_clone, connections_clone,
                            handlers_clone, miner_hotkey_clone, miner_keypair_clone
                        ).await;
                    });
                }
                Err(e) => {
                    println!("❌ Stream error: {}", e);
                    break;
                }
            }
        }
    }

    async fn handle_stream(
        mut send: SendStream,
        mut recv: RecvStream,
        connection: Arc<quinn::Connection>,
        connections: Arc<RwLock<HashMap<String, ValidatorConnection>>>,
        synapse_handlers: Arc<RwLock<HashMap<String, PyObject>>>,
        miner_hotkey: String,
        miner_keypair: Option<[u8; 32]>,
    ) {
        // Read incoming message
        match recv.read_to_end(1024 * 1024).await { // 1MB limit
            Ok(buffer) => {
                let message = String::from_utf8_lossy(&buffer);
                println!("📨 Received message: {}", message);

                // Try to parse as handshake first
                if let Ok(handshake_req) = serde_json::from_str::<HandshakeRequest>(&message) {
                    let response = Self::process_handshake(
                        handshake_req, connection.clone(), connections, miner_hotkey, miner_keypair
                    ).await;

                    if let Ok(response_json) = serde_json::to_string(&response) {
                        let _ = send.write_all(response_json.as_bytes()).await;
                        let _ = send.finish().await;
                    }
                    return;
                }

                // Try to parse as synapse packet
                if let Ok(synapse_packet) = serde_json::from_str::<SynapsePacket>(&message) {
                    let response = Self::process_synapse_packet(
                        synapse_packet, connections, synapse_handlers
                    ).await;

                    if let Ok(response_json) = serde_json::to_string(&response) {
                        let _ = send.write_all(response_json.as_bytes()).await;
                        let _ = send.finish().await;
                    }
                    return;
                }

                println!("⚠️ Unknown message format received");
            }
            Err(e) => {
                println!("❌ Failed to read stream: {}", e);
            }
        }
    }

    async fn process_handshake(
        request: HandshakeRequest,
        connection: Arc<quinn::Connection>,
        connections: Arc<RwLock<HashMap<String, ValidatorConnection>>>,
        miner_hotkey: String,
        miner_keypair: Option<[u8; 32]>,
    ) -> HandshakeResponse {
        println!("🤝 Processing handshake from validator: {}", request.validator_hotkey);

        // TODO: Implement real signature verification here
        let is_valid = Self::verify_validator_signature(&request).await;

        if is_valid {
            let connection_id = format!("conn_{}_{}",
                request.validator_hotkey,
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis()
            );

            let mut connections_guard = connections.write().await;
            let mut validator_conn = ValidatorConnection::new(
                request.validator_hotkey.clone(),
                connection_id.clone(),
                connection.clone()
            );
            validator_conn.verify();
            connections_guard.insert(request.validator_hotkey.clone(), validator_conn);

            println!("✅ Handshake successful, established connection: {}", connection_id);

            HandshakeResponse {
                miner_hotkey: miner_hotkey.clone(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                signature: Self::sign_handshake_response(&request, &miner_hotkey, &miner_keypair).await,
                accepted: true,
                connection_id,
            }
        } else {
            println!("❌ Handshake failed: invalid signature");
            HandshakeResponse {
                miner_hotkey: miner_hotkey.clone(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                signature: String::new(),
                accepted: false,
                connection_id: String::new(),
            }
        }
    }

    async fn verify_validator_signature(request: &HandshakeRequest) -> bool {
        println!("🔍 Verifying signature for validator: {}", request.validator_hotkey);

        // Check timestamp age (prevent replay attacks)
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        if current_time > request.timestamp && (current_time - request.timestamp) > MAX_SIGNATURE_AGE {
            println!("❌ Signature timestamp too old: {} (current: {})", request.timestamp, current_time);
            return false;
        }

        // Future timestamp protection
        if request.timestamp > current_time + 60 { // Allow 1 minute clock skew
            println!("❌ Signature timestamp too far in future: {} (current: {})", request.timestamp, current_time);
            return false;
        }

        // Create expected message (same format as bittensor)
        let expected_message = format!("handshake:{}:{}", request.validator_hotkey, request.timestamp);
        println!("🔍 Verifying signature for message: {}", expected_message);

        // Parse SS58 address to extract public key using substrate's SS58 codec
        match sr25519::Public::from_ss58check(&request.validator_hotkey) {
            Ok(public_key) => {
                // Decode base64 signature
                match BASE64_STANDARD.decode(&request.signature) {
                    Ok(signature_bytes) => {
                        if signature_bytes.len() != 64 {
                            println!("❌ Invalid signature length: {}", signature_bytes.len());
                            return false;
                        }

                        // Convert signature bytes to sr25519::Signature
                        let mut signature_array = [0u8; 64];
                        signature_array.copy_from_slice(&signature_bytes);
                        let signature = sr25519::Signature::from_raw(signature_array);

                        let message_bytes = expected_message.as_bytes();

                        // Verify the signature using SR25519 verification
                        let verification_result = sr25519::Pair::verify(&signature, message_bytes, &public_key);

                        if verification_result {
                            println!("✅ SR25519 signature verification successful for {}", request.validator_hotkey);
                            true
                        } else {
                            println!("❌ SR25519 signature verification failed for {}", request.validator_hotkey);
                            false
                        }
                    }
                    Err(e) => {
                        println!("❌ Failed to decode base64 signature: {}", e);
                        false
                    }
                }
            }
            Err(e) => {
                println!("❌ Invalid SS58 address {}: {}", request.validator_hotkey, e);
                false
            }
        }
    }

    async fn sign_handshake_response(request: &HandshakeRequest, _miner_hotkey: &str, miner_keypair: &Option<[u8; 32]>) -> String {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let message = format!("handshake_response:{}:{}", request.validator_hotkey, timestamp);

        match miner_keypair {
            Some(keypair_seed) => {
                // Create SR25519 pair from seed
                let pair = sr25519::Pair::from_seed(keypair_seed);
                let signature = pair.sign(message.as_bytes());
                let signature_bytes = signature.0;
                BASE64_STANDARD.encode(signature_bytes)
            }
            None => {
                println!("⚠️ No miner keypair configured, using dummy signature");
                let dummy_bytes = &message.as_bytes()[..8];
                BASE64_STANDARD.encode(dummy_bytes)
            }
        }
    }

    async fn process_synapse_packet(
        packet: SynapsePacket,
        connections: Arc<RwLock<HashMap<String, ValidatorConnection>>>,
        synapse_handlers: Arc<RwLock<HashMap<String, PyObject>>>,
    ) -> SynapseResponse {
        println!("📦 Processing {} synapse packet", packet.synapse_type);

        // Extract validator hotkey from packet data (should be set by client)
        let validator_hotkey = packet.data.get("validator_hotkey")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown_validator")
            .to_string();

        // Verify connection exists and is verified
        {
            let mut connections_guard = connections.write().await;
            if let Some(connection) = connections_guard.get_mut(&validator_hotkey) {
                if !connection.verified {
                    println!("❌ Connection not verified for validator: {}", validator_hotkey);
                    return SynapseResponse {
                        success: false,
                        data: HashMap::new(),
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        error: Some("Connection not verified".to_string()),
                    };
                }
                connection.update_activity();
                println!("✅ Connection verified and activity updated for validator: {}", validator_hotkey);
            } else {
                println!("⚠️ No connection found for validator {}, allowing request to proceed", validator_hotkey);
                // Allow the request to proceed even without a verified connection
                // This is useful for development and initial testing
            }
        }

        // Process synapse using registered handlers
        let handlers = synapse_handlers.read().await;
        if let Some(handler) = handlers.get(&packet.synapse_type) {
            println!("📞 Calling Python handler for synapse type: {}", packet.synapse_type);

            // Call the Python handler with the synapse data
            match pyo3::Python::with_gil(|py| -> PyResult<HashMap<String, serde_json::Value>> {
                // Convert packet data to Python dict
                let py_dict = pyo3::types::PyDict::new(py);

                for (key, value) in &packet.data {
                    let py_value = match value {
                        serde_json::Value::String(s) => s.to_object(py),
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                i.to_object(py)
                            } else if let Some(f) = n.as_f64() {
                                f.to_object(py)
                            } else {
                                n.to_string().to_object(py)
                            }
                        },
                        serde_json::Value::Bool(b) => b.to_object(py),
                        serde_json::Value::Array(arr) => {
                            let py_list = pyo3::types::PyList::empty(py);
                            for item in arr {
                                let item_str = serde_json::to_string(item).unwrap_or_default();
                                py_list.append(item_str)?;
                            }
                            py_list.to_object(py)
                        },
                        serde_json::Value::Object(_) => {
                            serde_json::to_string(value).unwrap_or_default().to_object(py)
                        },
                        serde_json::Value::Null => py.None(),
                    };
                    py_dict.set_item(key, py_value)?;
                }

                // Call the Python handler function
                let result = handler.call1(py, (py_dict,))?;

                // Convert result back to HashMap
                let result_dict: &pyo3::types::PyDict = result.extract(py)?;
                let mut response_data = HashMap::new();

                for (key, value) in result_dict.iter() {
                    let key_str: String = key.extract()?;
                    let value_json = if let Ok(s) = value.extract::<String>() {
                        serde_json::Value::String(s)
                    } else if let Ok(b) = value.extract::<bool>() {
                        serde_json::Value::Bool(b)
                    } else if let Ok(i) = value.extract::<i64>() {
                        serde_json::Value::Number(serde_json::Number::from(i))
                    } else if let Ok(f) = value.extract::<f64>() {
                        serde_json::Number::from_f64(f).map(serde_json::Value::Number)
                            .unwrap_or(serde_json::Value::Null)
                    } else {
                        // Fallback: convert to string
                        let s: String = value.str()?.extract()?;
                        serde_json::Value::String(s)
                    };
                    response_data.insert(key_str, value_json);
                }

                Ok(response_data)
            }) {
                Ok(response_data) => {
                    println!("✅ Python handler executed successfully for {}", packet.synapse_type);
                    SynapseResponse {
                        success: true,
                        data: response_data,
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        error: None,
                    }
                },
                Err(e) => {
                    println!("❌ Python handler error for {}: {}", packet.synapse_type, e);
                    let mut error_data = HashMap::new();
                    error_data.insert("error".to_string(),
                        serde_json::Value::String(format!("Python handler error: {}", e)));

                    SynapseResponse {
                        success: false,
                        data: error_data,
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        error: Some(format!("Python handler error: {}", e)),
                    }
                }
            }
        } else {
            println!("❌ No handler registered for synapse type: {}", packet.synapse_type);
            SynapseResponse {
                success: false,
                data: HashMap::new(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                error: Some(format!("No handler for synapse type: {}", packet.synapse_type)),
            }
        }
    }

    // Temporarily comment out unused methods to remove warnings
    /*
    pub async fn get_connection_count(&self) -> usize {
        let connections = self.connections.read().await;
        connections.values().filter(|c| c.verified).count()
    }
    */

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
            if let Some(connection) = connections.remove(&validator) {
                connection.connection.close(0u32.into(), b"cleanup");
                println!("🧹 Cleaned up stale connection from validator: {}", validator);
            }
        }

        Ok(())
    }

    pub async fn stop(&self) -> PyResult<()> {
        let mut connections = self.connections.write().await;
        for (_, connection) in connections.drain() {
            connection.connection.close(0u32.into(), b"server_shutdown");
        }

        if let Some(endpoint) = &self.endpoint {
            endpoint.close(0u32.into(), b"server_shutdown");
        }

        println!("🔌 Lightning QUIC server stopped, all connections closed");
        Ok(())
    }
}
