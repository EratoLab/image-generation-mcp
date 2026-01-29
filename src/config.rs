//! Configuration module for Image Generation MCP Server
//!
//! This module handles loading and managing configuration from TOML files.

use crate::error::{BrowserError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;

/// Global configuration storage
static GLOBAL_CONFIG: OnceLock<Arc<ServerConfig>> = OnceLock::new();

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub server: ServerSection,
    pub tool: ToolSection,
    pub metadata: MetadataConfig,
    #[serde(default)]
    pub transport: TransportConfig,
}

/// Server metadata section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSection {
    pub name: String,
    pub version: String,
}

/// Transport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConfig {
    /// Transport type: stdio, sse, or http
    #[serde(default = "default_transport_type")]
    pub transport: String,

    /// Host address to bind to
    #[serde(default = "default_host")]
    pub host: String,

    /// Port for SSE or HTTP transport
    #[serde(default = "default_port")]
    pub port: u16,

    /// SSE endpoint path
    #[serde(default = "default_sse_path")]
    pub sse_path: String,

    /// SSE POST path for messages
    #[serde(default = "default_sse_post_path")]
    pub sse_post_path: String,

    /// HTTP streamable endpoint path
    #[serde(default = "default_http_path")]
    pub http_path: String,
}

fn default_transport_type() -> String {
    "stdio".to_string()
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    3000
}

fn default_sse_path() -> String {
    "/sse".to_string()
}

fn default_sse_post_path() -> String {
    "/message".to_string()
}

fn default_http_path() -> String {
    "/mcp".to_string()
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            transport: default_transport_type(),
            host: default_host(),
            port: default_port(),
            sse_path: default_sse_path(),
            sse_post_path: default_sse_post_path(),
            http_path: default_http_path(),
        }
    }
}

/// Tool configuration with tagged variants for extensibility
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolSection {
    Mock(MockToolConfig),
    // Future backends can be added here:
    // OpenAI(OpenAIToolConfig),
    // StabilityAI(StabilityAIToolConfig),
}

/// Mock tool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockToolConfig {
    pub static_image_path: String,
}

/// Metadata configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataConfig {
    pub prefix: String,
}

impl ServerConfig {
    /// Load configuration from file path with priority fallback
    ///
    /// Priority order:
    /// 1. Explicit path provided
    /// 2. MCP_SERVER_CONFIG environment variable
    /// 3. ./mcp-server.toml (default)
    #[cfg(feature = "mcp-server")]
    pub async fn load(explicit_path: Option<String>) -> Result<Self> {
        let config_path = Self::resolve_config_path(explicit_path)?;
        Self::load_from_file(&config_path).await
    }

    /// Load configuration from file path with priority fallback (blocking version)
    ///
    /// Priority order:
    /// 1. Explicit path provided
    /// 2. MCP_SERVER_CONFIG environment variable
    /// 3. ./mcp-server.toml (default)
    #[cfg(not(feature = "mcp-server"))]
    pub async fn load(explicit_path: Option<String>) -> Result<Self> {
        let config_path = Self::resolve_config_path(explicit_path)?;
        let content = std::fs::read_to_string(&config_path)
            .map_err(|e| BrowserError::ConfigError(format!("Failed to read config file {}: {}", config_path.display(), e)))?;

        let config: ServerConfig = toml::from_str(&content)
            .map_err(|e| BrowserError::ConfigError(format!("Failed to parse config file {}: {}", config_path.display(), e)))?;

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Resolve the configuration file path based on priority
    fn resolve_config_path(explicit_path: Option<String>) -> Result<PathBuf> {
        // 1. Check explicit path (highest priority)
        if let Some(path) = explicit_path {
            return Ok(PathBuf::from(path));
        }

        // 2. Check environment variable
        if let Ok(path) = std::env::var("MCP_SERVER_CONFIG") {
            return Ok(PathBuf::from(path));
        }

        // 3. Use default path
        let default_path = PathBuf::from("./mcp-server.toml");

        // Check if default file exists
        if !default_path.exists() {
            return Err(BrowserError::ConfigError(format!(
                "Configuration file not found. Searched in order:\n\
                 1. Explicit path via --config argument\n\
                 2. MCP_SERVER_CONFIG environment variable\n\
                 3. ./mcp-server.toml (default)\n\n\
                 Please create a configuration file. Example:\n\
                 [server]\n\
                 name = \"image-generation-mcp\"\n\
                 version = \"0.1.0\"\n\n\
                 [tool]\n\
                 type = \"mock\"\n\n\
                 [tool.mock]\n\
                 static_image_path = \"static/image_1.png\"\n\n\
                 [metadata]\n\
                 prefix = \"chat.erato/\""
            )));
        }

        Ok(default_path)
    }

    /// Load configuration from a specific file
    #[cfg(feature = "mcp-server")]
    async fn load_from_file(path: &PathBuf) -> Result<Self> {
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| BrowserError::ConfigError(format!("Failed to read config file {}: {}", path.display(), e)))?;

        let config: ServerConfig = toml::from_str(&content)
            .map_err(|e| BrowserError::ConfigError(format!("Failed to parse config file {}: {}", path.display(), e)))?;

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Validate configuration
    fn validate(&self) -> Result<()> {
        // Validate metadata prefix
        if !self.metadata.prefix.ends_with('/') {
            return Err(BrowserError::ConfigError(
                "Metadata prefix must end with '/'".to_string()
            ));
        }

        // Validate transport type
        let valid_transports = ["stdio", "sse", "http"];
        if !valid_transports.contains(&self.transport.transport.as_str()) {
            return Err(BrowserError::ConfigError(format!(
                "Invalid transport type: '{}'. Must be one of: {}",
                self.transport.transport,
                valid_transports.join(", ")
            )));
        }

        // Validate tool-specific configuration
        match &self.tool {
            ToolSection::Mock(mock_config) => {
                if mock_config.static_image_path.is_empty() {
                    return Err(BrowserError::ConfigError(
                        "Mock tool static_image_path cannot be empty".to_string()
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Set the global configuration
pub async fn set_global_config(config: Arc<ServerConfig>) {
    let _ = GLOBAL_CONFIG.get_or_init(|| config).clone();
}

/// Get the global configuration
pub async fn get_global_config() -> Arc<ServerConfig> {
    GLOBAL_CONFIG
        .get()
        .expect("Global config not initialized. Call set_global_config() first.")
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_deserialize() {
        let toml_str = r#"
            [server]
            name = "image-generation-mcp"
            version = "0.1.0"

            [tool]
            type = "mock"
            static_image_path = "static/image_1.png"

            [metadata]
            prefix = "chat.erato/"

            [transport]
            transport = "stdio"
        "#;

        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.name, "image-generation-mcp");
        assert_eq!(config.server.version, "0.1.0");
        assert_eq!(config.metadata.prefix, "chat.erato/");
        assert_eq!(config.transport.transport, "stdio");

        match config.tool {
            ToolSection::Mock(mock) => {
                assert_eq!(mock.static_image_path, "static/image_1.png");
            }
        }
    }

    #[test]
    fn test_config_validate_prefix() {
        let toml_str = r#"
            [server]
            name = "image-generation-mcp"
            version = "0.1.0"

            [tool]
            type = "mock"
            static_image_path = "static/image_1.png"

            [metadata]
            prefix = "invalid"

            [transport]
            transport = "stdio"
        "#;

        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(config.validate().is_err());
    }
}
