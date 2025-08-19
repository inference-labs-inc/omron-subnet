use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use anyhow::Result;
use serde_json::Value;
use futures::future::join_all;

#[derive(Clone)]
struct AxonInfo {
    ip: String,
    port: u16,
    hotkey: String,
}

#[derive(Clone)]
struct SynapseRequest {
    headers: HashMap<String, String>,
    body: Value,
    synapse_name: String,
}

struct DendriteRequest {
    axon: AxonInfo,
    synapse: SynapseRequest,
    signature: String,
    url: String,
}

#[pyclass]
pub struct LightningDendrite {
    client: reqwest::Client,
    runtime: Arc<tokio::runtime::Runtime>,
    wallet_hotkey: String,
    external_ip: String,
    uuid: String,
}

#[pymethods]
impl LightningDendrite {
    #[new]
    #[pyo3(signature = (wallet_hotkey, external_ip=None))]
    fn new(wallet_hotkey: String, external_ip: Option<String>) -> Self {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("Failed to create Tokio runtime")
        );

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        let uuid = uuid::Uuid::new_v4().to_string();
        let external_ip = external_ip.unwrap_or_else(|| "127.0.0.1".to_string());

        Self {
            client,
            runtime,
            wallet_hotkey,
            external_ip,
            uuid,
        }
    }

    fn call<'py>(
        &self,
        py: Python<'py>,
        target_axon: &Bound<'py, PyDict>,
        synapse_headers: &Bound<'py, PyDict>,
        synapse_body: &Bound<'py, PyDict>,
        signature: String,
        timeout: f64,
    ) -> PyResult<PyObject> {
        let axon = parse_axon_info(target_axon)?;
        let synapse = parse_synapse_request(synapse_headers, synapse_body)?;

        let request = DendriteRequest {
            url: build_url(&self.external_ip, &axon, &synapse.synapse_name),
            axon,
            synapse,
            signature,
        };

        let client = self.client.clone();
        let timeout_duration = Duration::from_secs_f64(timeout);

        let future = async move {
            execute_request(client, request, timeout_duration).await
        };

        let result = self.runtime.block_on(future);

        match result {
            Ok(response) => {
                let response_dict = PyDict::new_bound(py);
                response_dict.set_item("status_code", response.status_code)?;
                response_dict.set_item("status_message", response.status_message)?;
                response_dict.set_item("response_data", response.response_data)?;
                response_dict.set_item("process_time", response.process_time)?;
                Ok(response_dict.into())
            }
            Err(e) => {
                let error_dict = PyDict::new_bound(py);
                error_dict.set_item("status_code", "500")?;
                error_dict.set_item("status_message", format!("Request failed: {}", e))?;
                error_dict.set_item("response_data", py.None())?;
                error_dict.set_item("process_time", "0.0")?;
                Ok(error_dict.into())
            }
        }
    }

    fn forward<'py>(
        &self,
        py: Python<'py>,
        axons: &Bound<'py, PyList>,
        synapse_headers: &Bound<'py, PyDict>,
        synapse_body: &Bound<'py, PyDict>,
        signatures: &Bound<'py, PyList>,
        timeout: f64,
    ) -> PyResult<PyObject> {
        let mut requests = Vec::new();

        for (i, axon_obj) in axons.iter().enumerate() {
            let axon_dict = axon_obj.downcast::<PyDict>()?;
            let axon = parse_axon_info(&axon_dict)?;
            let synapse = parse_synapse_request(synapse_headers, synapse_body)?;
            let signature: String = signatures.get_item(i)?.extract()?;

            let request = DendriteRequest {
                url: build_url(&self.external_ip, &axon, &synapse.synapse_name),
                axon,
                synapse,
                signature,
            };

            requests.push(request);
        }

        let client = self.client.clone();
        let timeout_duration = Duration::from_secs_f64(timeout);

        let future = async move {
            let futures: Vec<_> = requests
                .into_iter()
                .map(|req| execute_request(client.clone(), req, timeout_duration))
                .collect();

            join_all(futures).await
        };

        let results = self.runtime.block_on(future);
        let response_list = PyList::empty_bound(py);

        for result in results {
            let response_dict = PyDict::new_bound(py);

            match result {
                Ok(response) => {
                    response_dict.set_item("status_code", response.status_code)?;
                    response_dict.set_item("status_message", response.status_message)?;
                    response_dict.set_item("response_data", response.response_data)?;
                    response_dict.set_item("process_time", response.process_time)?;
                }
                Err(e) => {
                    response_dict.set_item("status_code", "500")?;
                    response_dict.set_item("status_message", format!("Request failed: {}", e))?;
                    response_dict.set_item("response_data", py.None())?;
                    response_dict.set_item("process_time", "0.0")?;
                }
            }

            response_list.append(response_dict)?;
        }

        Ok(response_list.into())
    }
}

#[derive(Debug)]
struct DendriteResponse {
    status_code: String,
    status_message: String,
    response_data: Option<String>,
    process_time: String,
}

async fn execute_request(
    client: reqwest::Client,
    request: DendriteRequest,
    timeout: Duration,
) -> Result<DendriteResponse> {
    let start_time = SystemTime::now();

    let mut headers_map = reqwest::header::HeaderMap::new();
    for (key, value) in &request.synapse.headers {
        headers_map.insert(
            reqwest::header::HeaderName::from_bytes(key.as_bytes())?,
            reqwest::header::HeaderValue::from_str(value)?,
        );
    }

    // Add signature to headers
    headers_map.insert(
        reqwest::header::HeaderName::from_static("dendrite-signature"),
        reqwest::header::HeaderValue::from_str(&request.signature)?,
    );

    let response_result = tokio::time::timeout(
        timeout,
        client
            .post(&request.url)
            .headers(headers_map)
            .json(&request.synapse.body)
            .send(),
    ).await;

    let process_time = start_time.elapsed().unwrap_or_default().as_secs_f64().to_string();

    match response_result {
        Ok(Ok(response)) => {
            let status_code = response.status().as_u16().to_string();
            let status_message = if response.status().is_success() {
                "Success".to_string()
            } else {
                format!("HTTP {}", response.status().as_u16())
            };

            let response_data = if response.status().is_success() {
                match response.text().await {
                    Ok(text) => Some(text),
                    Err(_) => None,
                }
            } else {
                None
            };

            Ok(DendriteResponse {
                status_code,
                status_message,
                response_data,
                process_time,
            })
        }
        Ok(Err(e)) => {
            let (status_code, status_message) = if e.is_timeout() {
                ("408".to_string(), "Request timeout".to_string())
            } else if e.is_connect() {
                ("503".to_string(), format!("Connection error: {}", e))
            } else {
                ("500".to_string(), format!("Request error: {}", e))
            };

            Ok(DendriteResponse {
                status_code,
                status_message,
                response_data: None,
                process_time,
            })
        }
        Err(_) => {
            Ok(DendriteResponse {
                status_code: "408".to_string(),
                status_message: "Request timeout".to_string(),
                response_data: None,
                process_time,
            })
        }
    }
}

fn parse_axon_info(axon_dict: &Bound<'_, PyDict>) -> PyResult<AxonInfo> {
    let ip: String = axon_dict.get_item("ip")?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'ip' field")
    })?.extract()?;

    let port: u16 = axon_dict.get_item("port")?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'port' field")
    })?.extract()?;

    let hotkey: String = axon_dict.get_item("hotkey")?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'hotkey' field")
    })?.extract()?;

    Ok(AxonInfo { ip, port, hotkey })
}

fn parse_synapse_request(headers: &Bound<'_, PyDict>, body: &Bound<'_, PyDict>) -> PyResult<SynapseRequest> {
    let mut header_map = HashMap::new();
    for (key, value) in headers.iter() {
        let key_str: String = key.extract()?;
        let value_str: String = value.extract()?;
        header_map.insert(key_str, value_str);
    }

    // Convert PyDict to serde_json::Value manually
    let mut body_map = serde_json::Map::new();
    for (key, value) in body.iter() {
        let key_str: String = key.extract()?;
        // For simplicity, convert all values to strings - could be enhanced for type preservation
        let value_str: String = if value.is_none() {
            String::new()
        } else {
            format!("{}", value)
        };
        body_map.insert(key_str, Value::String(value_str));
    }
    let body_value = Value::Object(body_map);

    // Extract synapse name from headers or use default
    let synapse_name = header_map
        .get("synapse-name")
        .cloned()
        .unwrap_or_else(|| "Synapse".to_string());

    Ok(SynapseRequest {
        headers: header_map,
        body: body_value,
        synapse_name,
    })
}

fn build_url(external_ip: &str, axon: &AxonInfo, synapse_name: &str) -> String {
    let endpoint_ip = if axon.ip == external_ip || axon.ip == "0.0.0.0" {
        "127.0.0.1"
    } else {
        &axon.ip
    };

    format!("http://{}:{}/{}", endpoint_ip, axon.port, synapse_name)
}

#[pymodule]
fn lightning(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LightningDendrite>()?;
    Ok(())
}
