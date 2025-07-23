use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Connection {
    pub id: String,
    pub endpoint: String,
    pub created_at: u64,
}

impl Connection {
    pub fn new(id: String, endpoint: String) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        Self {
            id,
            endpoint,
            created_at: now,
        }
    }
}

#[derive(Debug)]
pub struct ConnectionPool {
    connections: Arc<RwLock<HashMap<String, Connection>>>,
}

impl ConnectionPool {
    pub fn new() -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn add_connection(&mut self, endpoint: &str, connection_id: String) {
        let mut connections = self.connections.write().await;
        let connection = Connection::new(connection_id, endpoint.to_string());
        connections.insert(endpoint.to_string(), connection);
        println!("🔗 Added persistent connection to pool: {}", endpoint);
    }

    pub async fn get_connection(&self, endpoint: &str) -> Option<String> {
        let connections = self.connections.read().await;
        connections.get(endpoint).map(|c| c.id.clone())
    }

    pub async fn remove_connection(&mut self, endpoint: &str) {
        let mut connections = self.connections.write().await;
        if let Some(_connection) = connections.remove(endpoint) {
            println!("🔌 Removed connection from pool: {}", endpoint);
        }
    }

    pub async fn connection_count(&self) -> usize {
        let connections = self.connections.read().await;
        connections.len()
    }

    pub async fn close_all(&mut self) {
        let mut connections = self.connections.write().await;
        connections.clear();
        println!("🔌 Closed all connections in pool");
    }

    // Temporarily comment out unused methods to remove warnings
    /*
    pub async fn get_all_connections(&self) -> Vec<(String, String)> {
        let connections = self.connections.read().await;
        connections.iter()
            .map(|(endpoint, conn)| (endpoint.clone(), conn.id.clone()))
            .collect()
    }

    pub async fn cleanup_stale_connections(&mut self, max_age_seconds: u64) {
        // Implementation would go here
    }

    pub async fn get_connection_stats(&self) -> HashMap<String, u64> {
        // Implementation would go here
        HashMap::new()
    }
    */
}
