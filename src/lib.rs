//! # image-generation-mcp
//!
//! A Model Context Protocol (MCP) server for image generation with configurable backends.
//!
//! ## Features
//!
//! - **MCP Server**: Model Context Protocol server for image generation
//! - **Configurable Backends**: Switch between image generation providers via TOML configuration
//! - **Token Usage Tracking**: Built-in tracking with `_meta` fields
//! - **Simple Configuration**: Configure via TOML file or environment variables
//!
//! ## MCP Server
//!
//! ### Running the MCP Server
//!
//! ```bash
//! # Using default config path (./mcp-server.toml)
//! cargo run --bin mcp-server
//!
//! # Using custom config path
//! cargo run --bin mcp-server -- --config /path/to/config.toml
//!
//! # Using environment variable
//! export MCP_SERVER_CONFIG=/path/to/config.toml
//! cargo run --bin mcp-server
//! ```
//!
//! ## Module Overview
//!
//! - [`config`]: Configuration management
//! - [`tools`]: Image generation tools
//! - [`error`]: Error types and result aliases
//! - [`mcp`]: **Model Context Protocol server** (requires `mcp-handler` feature)

pub mod config;
pub mod error;
pub mod tools;

#[cfg(feature = "mcp-handler")]
pub mod mcp;

pub use error::{BrowserError, Result};
pub use tools::{Tool, ToolContext, ToolRegistry, ToolResult};

#[cfg(feature = "mcp-handler")]
pub use mcp::BrowserServer;
#[cfg(feature = "mcp-handler")]
pub use rmcp::ServiceExt;
