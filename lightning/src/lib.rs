use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

mod client;
mod server;
mod types;
mod connection_pool;
mod serialization;

use client::LightningClient;
use server::LightningServer;
use types::{QuicAxonInfo, QuicRequest};

#[pyclass]
pub struct RustLightning {
    client: Mutex<LightningClient>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl RustLightning {
    #[new]
    pub fn new(wallet_hotkey: String) -> PyResult<Self> {
        let runtime = Arc::new(tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create async runtime: {}", e)
            ))?);

        let client = LightningClient::new(wallet_hotkey);

        Ok(Self {
            client: Mutex::new(client),
            runtime
        })
    }

    pub fn set_validator_keypair(&self, keypair_seed: [u8; 32]) -> PyResult<()> {
        let mut client = self.client.lock().unwrap();
        client.set_validator_keypair(keypair_seed);
        Ok(())
    }

    pub fn initialize_connections(&self, miners: Vec<PyObject>) -> PyResult<()> {
        pyo3::Python::with_gil(|py| {
            let mut quic_miners = Vec::new();

            for miner_obj in miners {
                let miner_dict = miner_obj.extract::<HashMap<String, PyObject>>(py)?;

                let hotkey = miner_dict.get("hotkey")
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'hotkey' field"))?
                    .extract::<String>(py)?;

                let ip = miner_dict.get("ip")
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'ip' field"))?
                    .extract::<String>(py)?;

                let port = miner_dict.get("port")
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'port' field"))?
                    .extract::<u16>(py)?;

                let protocol = miner_dict.get("protocol").map(|p| p.extract::<u8>(py)).transpose()?.unwrap_or(4);
                let placeholder1 = miner_dict.get("placeholder1").map(|p| p.extract::<u8>(py)).transpose()?.unwrap_or(0);
                let placeholder2 = miner_dict.get("placeholder2").map(|p| p.extract::<u8>(py)).transpose()?.unwrap_or(0);

                quic_miners.push(QuicAxonInfo::new(hotkey, ip, port, protocol, placeholder1, placeholder2));
            }

            let mut client = self.client.lock().unwrap();
            self.runtime.block_on(async {
                client.initialize_connections(quic_miners).await
            })
        })
    }

    pub fn query_axon(&self, axon_data: PyObject, request_data: PyObject) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let axon_dict = axon_data.extract::<HashMap<String, PyObject>>(py)?;
            let request_dict = request_data.extract::<HashMap<String, PyObject>>(py)?;

            let hotkey = axon_dict.get("hotkey")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'hotkey' field"))?
                .extract::<String>(py)?;

            let ip = axon_dict.get("ip")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'ip' field"))?
                .extract::<String>(py)?;

            let port = axon_dict.get("port")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'port' field"))?
                .extract::<u16>(py)?;

            let protocol = axon_dict.get("protocol").map(|p| p.extract::<u8>(py)).transpose()?.unwrap_or(4);
            let placeholder1 = axon_dict.get("placeholder1").map(|p| p.extract::<u8>(py)).transpose()?.unwrap_or(0);
            let placeholder2 = axon_dict.get("placeholder2").map(|p| p.extract::<u8>(py)).transpose()?.unwrap_or(0);

            let axon_info = QuicAxonInfo::new(hotkey, ip, port, protocol, placeholder1, placeholder2);

            let synapse_type = request_dict.get("synapse_type")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'synapse_type' field"))?
                .extract::<String>(py)?;

            let mut data = HashMap::new();
            if let Some(data_obj) = request_dict.get("data") {
                let data_dict = data_obj.extract::<HashMap<String, PyObject>>(py)?;
                for (key, value) in data_dict {
                    let json_str = format!("{}", value.as_ref(py).repr()?);
                    let json_value: serde_json::Value = serde_json::from_str(&json_str)
                        .unwrap_or_else(|_| serde_json::Value::String(value.extract::<String>(py).unwrap_or_default()));
                    data.insert(key, json_value);
                }
            }

            let request = QuicRequest::new(synapse_type, data);

            let client = self.client.lock().unwrap();
            let response = self.runtime.block_on(async {
                client.query_axon(axon_info, request).await
            })?;

            let result_dict = pyo3::types::PyDict::new(py);
            result_dict.set_item("success", response.success)?;
            result_dict.set_item("latency_ms", response.latency_ms)?;

            for (key, value) in response.data {
                let py_value = match value {
                    serde_json::Value::String(s) => s.into_py(py),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            i.into_py(py)
                        } else if let Some(f) = n.as_f64() {
                            f.into_py(py)
                        } else {
                            n.to_string().into_py(py)
                        }
                    },
                    serde_json::Value::Bool(b) => b.into_py(py),
                    _ => serde_json::to_string(&value).unwrap_or_default().into_py(py),
                };
                result_dict.set_item(key, py_value)?;
            }

            Ok(result_dict.into())
        })
    }

    pub fn update_miner_registry(&self, miners: Vec<PyObject>) -> PyResult<()> {
        pyo3::Python::with_gil(|py| {
            let mut quic_miners = Vec::new();

            for miner_obj in miners {
                let miner_dict = miner_obj.extract::<HashMap<String, PyObject>>(py)?;

                let hotkey = miner_dict.get("hotkey")
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'hotkey' field"))?
                    .extract::<String>(py)?;

                let ip = miner_dict.get("ip")
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'ip' field"))?
                    .extract::<String>(py)?;

                let port = miner_dict.get("port")
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'port' field"))?
                    .extract::<u16>(py)?;

                let protocol = miner_dict.get("protocol").map(|p| p.extract::<u8>(py)).transpose()?.unwrap_or(4);
                let placeholder1 = miner_dict.get("placeholder1").map(|p| p.extract::<u8>(py)).transpose()?.unwrap_or(0);
                let placeholder2 = miner_dict.get("placeholder2").map(|p| p.extract::<u8>(py)).transpose()?.unwrap_or(0);

                quic_miners.push(QuicAxonInfo::new(hotkey, ip, port, protocol, placeholder1, placeholder2));
            }

            let mut client = self.client.lock().unwrap();
            self.runtime.block_on(async {
                client.update_miner_registry(quic_miners).await
            })
        })
    }

    pub fn get_connection_stats(&self) -> PyResult<PyObject> {
        let client = self.client.lock().unwrap();
        Python::with_gil(|py| {
            let stats = self.runtime.block_on(async {
                client.get_connection_stats().await
            })?;

            let result_dict = pyo3::types::PyDict::new(py);
            for (key, value) in stats {
                result_dict.set_item(key, value)?;
            }

            Ok(result_dict.into())
        })
    }

    pub fn close_all_connections(&self) -> PyResult<()> {
        let client = self.client.lock().unwrap();
        self.runtime.block_on(async {
            client.close_all_connections().await
        })
    }
}

#[pyclass]
pub struct RustLightningServer {
    server: Mutex<LightningServer>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl RustLightningServer {
    #[new]
    pub fn new(miner_hotkey: String, host: String, port: u16) -> PyResult<Self> {
        let runtime = Arc::new(tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create async runtime: {}", e)
            ))?);

        let server = LightningServer::new(miner_hotkey, host, port);

        Ok(Self {
            server: Mutex::new(server),
            runtime
        })
    }

    pub fn register_synapse_handler(&self, synapse_type: String, handler: PyObject) -> PyResult<()> {
        let server = self.server.lock().unwrap();
        self.runtime.block_on(async {
            server.register_synapse_handler(synapse_type, handler).await
        })
    }

    pub fn start(&self) -> PyResult<()> {
        let mut server = self.server.lock().unwrap();
        self.runtime.block_on(async {
            server.start().await
        })
    }

    pub fn serve_forever(&self) -> PyResult<()> {
        let mut server = self.server.lock().unwrap();
        self.runtime.block_on(async {
            server.serve_forever().await
        })
    }

    pub fn handle_handshake(&self, _handshake_data: PyObject) -> PyResult<PyObject> {
        // Placeholder for handshake handling
        Ok(pyo3::Python::with_gil(|py| py.None()))
    }

    pub fn handle_synapse_packet(&self, synapse_type: String, _packet_data: PyObject) -> PyResult<PyObject> {
        println!("📦 Handling synapse packet: {}", synapse_type);
        // Placeholder for synapse packet handling
        Ok(pyo3::Python::with_gil(|py| py.None()))
    }

    pub fn get_connection_stats(&self) -> PyResult<PyObject> {
        let server = self.server.lock().unwrap();
        Python::with_gil(|py| {
            let stats = self.runtime.block_on(async {
                server.get_connection_stats().await
            })?;

            let result_dict = pyo3::types::PyDict::new(py);
            for (key, value) in stats {
                result_dict.set_item(key, value)?;
            }

            Ok(result_dict.into())
        })
    }

    pub fn cleanup_stale_connections(&self, max_idle_seconds: u64) -> PyResult<()> {
        let server = self.server.lock().unwrap();
        self.runtime.block_on(async {
            server.cleanup_stale_connections(max_idle_seconds).await
        })
    }

    pub fn stop(&self) -> PyResult<()> {
        let server = self.server.lock().unwrap();
        self.runtime.block_on(async {
            server.stop().await
        })
    }
}

#[pymodule]
fn lightning(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustLightning>()?;
    m.add_class::<RustLightningServer>()?;
    m.add_class::<QuicAxonInfo>()?;
    Ok(())
}
