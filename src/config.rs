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
    /// Transport type: stdio or http
    #[serde(default = "default_transport_type")]
    pub transport: String,

    /// Host address to bind to
    #[serde(default = "default_host")]
    pub host: String,

    /// Port for HTTP transport
    #[serde(default = "default_port")]
    pub port: u16,

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

fn default_http_path() -> String {
    "/mcp".to_string()
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            transport: default_transport_type(),
            host: default_host(),
            port: default_port(),
            http_path: default_http_path(),
        }
    }
}

/// Tool configuration with tagged variants for extensibility
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolSection {
    Mock(MockToolConfig),
    GenAi(GenAiToolConfig),
    // Future backends can be added here:
    // OpenAI(OpenAIToolConfig),
    // StabilityAI(StabilityAIToolConfig),
}

/// Mock tool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockToolConfig {
    pub static_image_path: String,
}

/// GenAI-backed tool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenAiToolConfig {
    /// openai, azure_openai, or ollama
    pub provider_kind: String,
    pub model_name: String,
    pub model_display_name: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_version: Option<String>,
    pub additional_request_parameters: Option<Vec<String>>,
    pub additional_request_headers: Option<Vec<String>>,
    pub image_quality: Option<String>,
    pub image_style: Option<String>,
    pub response_format: Option<String>,
}

impl GenAiToolConfig {
    /// Parses additional_request_parameters into key/value pairs.
    pub fn additional_request_parameters_map(&self) -> std::collections::HashMap<String, String> {
        let mut params = std::collections::HashMap::new();
        if let Some(param_vec) = &self.additional_request_parameters {
            for param in param_vec {
                if let Some((key, value)) = param.split_once('=') {
                    params.insert(key.trim().to_string(), value.trim().to_string());
                }
            }
        }
        params
    }

    /// Parses additional_request_headers into key/value pairs.
    pub fn additional_request_headers_map(&self) -> std::collections::HashMap<String, String> {
        let mut headers = std::collections::HashMap::new();
        if let Some(header_vec) = &self.additional_request_headers {
            for header in header_vec {
                if let Some((key, value)) = header.split_once('=') {
                    headers.insert(key.trim().to_string(), value.trim().to_string());
                }
            }
        }
        headers
    }

    /// Migrates azure_openai configuration to openai format.
    pub fn migrate_azure_openai_to_openai(self) -> Result<Self> {
        if self.provider_kind != "azure_openai" {
            return Ok(self);
        }

        let base_url = if let Some(url) = &self.base_url {
            let trimmed_url = url.trim_end_matches('/');

            if !trimmed_url.contains(".api.cognitive.microsoft.com")
                && !trimmed_url.contains(".openai.azure.com")
                && !trimmed_url.contains(".cognitiveservices.azure.com")
                && !trimmed_url.contains(".services.ai.azure.com")
            {
                return Err(BrowserError::ConfigError(format!(
                    "Azure OpenAI base_url must end with either '.api.cognitive.microsoft.com', '.openai.azure.com', '.cognitiveservices.azure.com', or '.services.ai.azure.com' got: {}",
                    url
                )));
            }

            Some(format!(
                "{}/openai/deployments/{}/",
                trimmed_url, self.model_name
            ))
        } else {
            return Err(BrowserError::ConfigError(
                "base_url is required for azure_openai provider".to_string(),
            ));
        };

        let mut additional_params = self.additional_request_parameters.unwrap_or_default();
        if let Some(api_version) = &self.api_version {
            additional_params.push(format!("api-version={}", api_version));
        }

        let mut additional_headers = self.additional_request_headers.unwrap_or_default();
        if let Some(api_key) = &self.api_key {
            additional_headers.push(format!("api-key={}", api_key));
        }

        Ok(GenAiToolConfig {
            provider_kind: "openai".to_string(),
            model_name: self.model_name,
            model_display_name: self.model_display_name,
            base_url,
            api_key: None,
            api_version: None,
            additional_request_parameters: if additional_params.is_empty() {
                None
            } else {
                Some(additional_params)
            },
            additional_request_headers: if additional_headers.is_empty() {
                None
            } else {
                Some(additional_headers)
            },
            image_quality: self.image_quality,
            image_style: self.image_style,
            response_format: self.response_format,
        })
    }
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

        let config = config.migrate_tool_config()?;

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

        let config = config.migrate_tool_config()?;

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
        let valid_transports = ["stdio", "http"];
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
            ToolSection::GenAi(genai_config) => {
                let valid_providers = ["openai", "azure_openai", "ollama"];
                if !valid_providers.contains(&genai_config.provider_kind.as_str()) {
                    return Err(BrowserError::ConfigError(format!(
                        "Invalid genai provider_kind: '{}'. Must be one of: {}",
                        genai_config.provider_kind,
                        valid_providers.join(", ")
                    )));
                }
                if genai_config.model_name.trim().is_empty() {
                    return Err(BrowserError::ConfigError(
                        "GenAI tool model_name cannot be empty".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Migrate tool configuration for backwards-compatible provider aliases.
    fn migrate_tool_config(mut self) -> Result<Self> {
        if let ToolSection::GenAi(genai_config) = self.tool.clone() {
            self.tool = ToolSection::GenAi(genai_config.migrate_azure_openai_to_openai()?);
        }
        Ok(self)
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
            ToolSection::GenAi(_) => panic!("Expected mock tool config"),
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
