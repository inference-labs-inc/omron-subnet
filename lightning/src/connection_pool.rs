use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Connection {
    pub id: String,
    pub endpoint: String,
    pub created_at: u64,
    pub last_used: u64,
    pub active: bool,
}

impl Connection {
    pub fn new(id: String, endpoint: String) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        Self {
            id,
            endpoint,
            created_at: now,
            last_used: now,
            active: true,
        }
    }

    pub fn update_last_used(&mut self) {
        self.last_used = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }

    pub fn close(&mut self) {
        self.active = false;
    }
}

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
        let mut connections = self.connections.write().await;
        if let Some(connection) = connections.get_mut(endpoint) {
            if connection.active {
                connection.update_last_used();
                return Some(connection.id.clone());
            }
        }
        None
    }

    pub async fn remove_connection(&mut self, endpoint: &str) {
        let mut connections = self.connections.write().await;
        if let Some(mut connection) = connections.remove(endpoint) {
            connection.close();
            println!("🔌 Removed connection from pool: {}", endpoint);
        }
    }

    pub async fn connection_count(&self) -> usize {
        let connections = self.connections.read().await;
        connections.values().filter(|c| c.active).count()
    }

    pub async fn get_all_connections(&self) -> Vec<(String, String)> {
        let connections = self.connections.read().await;
        connections.iter()
            .filter(|(_, conn)| conn.active)
            .map(|(endpoint, conn)| (endpoint.clone(), conn.id.clone()))
            .collect()
    }

    pub async fn cleanup_stale_connections(&mut self, max_age_seconds: u64) {
        let mut connections = self.connections.write().await;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        let mut to_remove = Vec::new();
        for (endpoint, connection) in connections.iter_mut() {
            if now - connection.last_used > max_age_seconds {
                connection.close();
                to_remove.push(endpoint.clone());
            }
        }

        for endpoint in to_remove {
            connections.remove(&endpoint);
            println!("🧹 Cleaned up stale connection: {}", endpoint);
        }
    }

    pub async fn close_all(&mut self) {
        let mut connections = self.connections.write().await;
        for (_, connection) in connections.iter_mut() {
            connection.close();
        }
        connections.clear();
        println!("🔌 Closed all connections in pool");
    }

    pub async fn get_connection_stats(&self) -> HashMap<String, u64> {
        let connections = self.connections.read().await;
        let mut stats = HashMap::new();

        let active_count = connections.values().filter(|c| c.active).count() as u64;
        let total_count = connections.len() as u64;

        stats.insert("active_connections".to_string(), active_count);
        stats.insert("total_connections".to_string(), total_count);

        if !connections.is_empty() {
            let avg_age = connections.values()
                .map(|c| SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() - c.created_at)
                .sum::<u64>() / total_count;
            stats.insert("average_age_seconds".to_string(), avg_age);
        }

        stats
    }
}
