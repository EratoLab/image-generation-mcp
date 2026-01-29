//! ServerHandler implementation for ImageGenerationServer

use log::debug;
use rmcp::{
    ServerHandler,
    handler::server::tool::ToolRouter,
    model::{ServerCapabilities, ServerInfo},
    tool_handler,
};

/// MCP Server wrapper for image generation
///
/// This struct provides image generation capabilities via configurable backends
#[derive(Clone)]
pub struct BrowserServer {
    tool_router: ToolRouter<Self>,
}

impl BrowserServer {
    /// Create a new image generation server
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            tool_router: Self::tool_router(),
        })
    }
}

impl Default for BrowserServer {
    fn default() -> Self {
        Self::new().expect("Failed to create default image generation server")
    }
}

impl Drop for BrowserServer {
    fn drop(&mut self) {
        debug!("ImageGenerationServer dropped");
    }
}

#[tool_handler]
impl ServerHandler for BrowserServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Image Generation MCP Server - Generate images from text prompts using configurable backends".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}
