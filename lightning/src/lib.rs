use pyo3::prelude::*;
use std::collections::HashMap;
use tokio::runtime::Runtime;

mod types;
mod client;
mod server;
mod connection_pool;
mod serialization;

use types::{QuicAxonInfo, QuicRequest, QuicResponse, HandshakeRequest, SynapsePacket};
use client::LightningClient;
use server::LightningServer;

#[pyclass]
pub struct RustLightning {
    client: LightningClient,
    runtime: Runtime,
}

#[pymethods]
impl RustLightning {
    #[new]
    pub fn new(wallet_hotkey: String) -> PyResult<Self> {
        let runtime = Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create async runtime: {}", e)
            ))?;

        let client = LightningClient::new(wallet_hotkey);

        Ok(Self { client, runtime })
    }

    pub fn initialize_connections(&self, miners_data: Vec<PyObject>) -> PyResult<()> {
        Python::with_gil(|py| {
            let mut miners = Vec::new();

            for miner_obj in miners_data {
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

                miners.push(QuicAxonInfo::new(hotkey, ip, port, protocol, placeholder1, placeholder2));
            }

            self.runtime.block_on(async {
                self.client.initialize_connections(miners).await
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

            let response = self.runtime.block_on(async {
                self.client.query_axon(axon_info, request).await
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

    pub fn update_miner_registry(&self, miners_data: Vec<PyObject>) -> PyResult<()> {
        Python::with_gil(|py| {
            let mut miners = Vec::new();

            for miner_obj in miners_data {
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

                miners.push(QuicAxonInfo::new(hotkey, ip, port, protocol, placeholder1, placeholder2));
            }

            self.runtime.block_on(async {
                self.client.update_miner_registry(miners).await
            })
        })
    }

    pub fn get_connection_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = self.runtime.block_on(async {
                self.client.get_connection_stats().await
            })?;

            let result_dict = pyo3::types::PyDict::new(py);
            for (key, value) in stats {
                result_dict.set_item(key, value)?;
            }

            Ok(result_dict.into())
        })
    }

    pub fn close_all_connections(&self) -> PyResult<()> {
        self.runtime.block_on(async {
            self.client.close_all_connections().await
        })
    }
}

#[pyclass]
pub struct RustLightningServer {
    server: LightningServer,
    runtime: Runtime,
}

#[pymethods]
impl RustLightningServer {
    #[new]
    pub fn new(miner_hotkey: String, host: String, port: u16) -> PyResult<Self> {
        let runtime = Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create async runtime: {}", e)
            ))?;

        let server = LightningServer::new(miner_hotkey, host, port);

        Ok(Self { server, runtime })
    }

    pub fn register_synapse_handler(&self, synapse_type: String, handler: PyObject) -> PyResult<()> {
        self.runtime.block_on(async {
            self.server.register_synapse_handler(synapse_type, handler).await
        })
    }

    pub fn start(&self) -> PyResult<()> {
        self.runtime.block_on(async {
            self.server.start().await
        })
    }

    pub fn handle_handshake(&self, handshake_data: PyObject) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let handshake_dict = handshake_data.extract::<HashMap<String, PyObject>>(py)?;

            let validator_hotkey = handshake_dict.get("validator_hotkey")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'validator_hotkey' field"))?
                .extract::<String>(py)?;

            let timestamp = handshake_dict.get("timestamp")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'timestamp' field"))?
                .extract::<u64>(py)?;

            let signature = handshake_dict.get("signature")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'signature' field"))?
                .extract::<String>(py)?;

            let request = HandshakeRequest {
                validator_hotkey,
                timestamp,
                signature,
            };

            let response = self.runtime.block_on(async {
                self.server.handle_handshake(request).await
            })?;

            let result_dict = pyo3::types::PyDict::new(py);
            result_dict.set_item("miner_hotkey", response.miner_hotkey)?;
            result_dict.set_item("timestamp", response.timestamp)?;
            result_dict.set_item("signature", response.signature)?;
            result_dict.set_item("accepted", response.accepted)?;
            result_dict.set_item("connection_id", response.connection_id)?;

            Ok(result_dict.into())
        })
    }

    pub fn handle_synapse_packet(&self, validator_hotkey: String, packet_data: PyObject) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let packet_dict = packet_data.extract::<HashMap<String, PyObject>>(py)?;

            let synapse_type = packet_dict.get("synapse_type")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'synapse_type' field"))?
                .extract::<String>(py)?;

            let timestamp = packet_dict.get("timestamp")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'timestamp' field"))?
                .extract::<u64>(py)?;

            let mut data = HashMap::new();
            if let Some(data_obj) = packet_dict.get("data") {
                let data_dict = data_obj.extract::<HashMap<String, PyObject>>(py)?;
                for (key, value) in data_dict {
                    let json_str = format!("{}", value.as_ref(py).repr()?);
                    let json_value: serde_json::Value = serde_json::from_str(&json_str)
                        .unwrap_or_else(|_| serde_json::Value::String(value.extract::<String>(py).unwrap_or_default()));
                    data.insert(key, json_value);
                }
            }

            let packet = SynapsePacket {
                synapse_type,
                data,
                timestamp,
            };

            let response = self.runtime.block_on(async {
                self.server.handle_synapse_packet(validator_hotkey, packet).await
            })?;

            let result_dict = pyo3::types::PyDict::new(py);
            result_dict.set_item("success", response.success)?;
            result_dict.set_item("timestamp", response.timestamp)?;

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

            if let Some(error) = response.error {
                result_dict.set_item("error", error)?;
            }

            Ok(result_dict.into())
        })
    }

    pub fn get_connection_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = self.runtime.block_on(async {
                self.server.get_connection_stats().await
            })?;

            let result_dict = pyo3::types::PyDict::new(py);
            for (key, value) in stats {
                result_dict.set_item(key, value)?;
            }

            Ok(result_dict.into())
        })
    }

    pub fn cleanup_stale_connections(&self, max_idle_seconds: u64) -> PyResult<()> {
        self.runtime.block_on(async {
            self.server.cleanup_stale_connections(max_idle_seconds).await
        })
    }

    pub fn stop(&self) -> PyResult<()> {
        self.runtime.block_on(async {
            self.server.stop().await
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
