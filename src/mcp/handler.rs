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
        Self::new_with_tool_overrides(None, None, None, None, false)
    }

    /// Create a new image generation server with optional runtime tool metadata overrides.
    pub fn new_with_tool_overrides(
        tool_name: Option<String>,
        tool_description: Option<String>,
        edit_tool_name: Option<String>,
        edit_tool_description: Option<String>,
        edit_tool_enabled: bool,
    ) -> Result<Self, String> {
        let mut tool_router = Self::tool_router();
        if let Some(mut route) = tool_router.map.remove("generate_image") {
            if let Some(name) = tool_name {
                let trimmed = name.trim();
                if trimmed.is_empty() {
                    return Err("tool_name cannot be empty".to_string());
                }
                route.attr.name = trimmed.to_string().into();
            }

            if let Some(description) = tool_description {
                route.attr.description = Some(description.into());
            }

            let new_name = route.attr.name.to_string();
            tool_router.map.insert(new_name.into(), route);
        }

        if let Some(mut edit_route) = tool_router.map.remove("edit_image") {
            if !edit_tool_enabled {
                // Explicitly remove when feature flag is disabled.
                return Ok(Self { tool_router });
            }

            if let Some(name) = edit_tool_name {
                let trimmed = name.trim();
                if trimmed.is_empty() {
                    return Err("edit_tool_name cannot be empty".to_string());
                }
                edit_route.attr.name = trimmed.to_string().into();
            }

            if let Some(description) = edit_tool_description {
                edit_route.attr.description = Some(description.into());
            }

            let new_name = edit_route.attr.name.to_string();
            tool_router.map.insert(new_name.into(), edit_route);
        } else if edit_tool_enabled {
            return Err("edit_image tool override requested, but edit_image is not available in this build".to_string());
        }

        Ok(Self { tool_router })
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
