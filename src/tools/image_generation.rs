//! Image generation tool with mock implementation
//!
//! This module provides a mock image generation tool that returns a static image.
//! Future versions will support multiple backends (OpenAI, Stability AI, etc.)

use crate::config::ToolSection;
use crate::error::Result;
use crate::tools::{Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[cfg(feature = "mcp-server")]
use crate::error::BrowserError;

#[cfg(feature = "mcp-server")]
use base64::Engine;

#[cfg(feature = "mcp-server")]
use std::path::Path;

/// Image generation tool
#[derive(Debug, Default)]
pub struct ImageGenerationTool;

/// Parameters for image generation
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImageGenerationParams {
    /// Text prompt for image generation
    pub prompt: String,

    /// Optional width of the generated image (default: 1024)
    #[serde(default = "default_width")]
    pub width: Option<u32>,

    /// Optional height of the generated image (default: 1024)
    #[serde(default = "default_height")]
    pub height: Option<u32>,

    /// Number of images to generate (default: 1)
    #[serde(default = "default_num_images")]
    pub num_images: Option<u32>,
}

fn default_width() -> Option<u32> {
    Some(1024)
}

fn default_height() -> Option<u32> {
    Some(1024)
}

fn default_num_images() -> Option<u32> {
    Some(1)
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_tokens: u64,
    pub model_identifier: String,
}

/// Generated image response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    pub data_uri: String,
    pub width: u32,
    pub height: u32,
    pub mime_type: String,
}

/// Response wrapper for generated images
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    pub images: Vec<GeneratedImage>,
}

#[async_trait]
impl Tool for ImageGenerationTool {
    type Params = ImageGenerationParams;

    fn name(&self) -> &str {
        "generate_image"
    }

    async fn execute_typed(&self, params: Self::Params, _context: &mut ToolContext<'_>) -> Result<ToolResult> {
        // Calculate token usage (mock approximation)
        let token_usage = TokenUsage {
            total_prompt_tokens: (params.prompt.len() as u64) * 4,
            total_completion_tokens: 0,
            total_tokens: (params.prompt.len() as u64) * 4,
            model_identifier: "mock/v1".to_string(),
        };

        // Execute based on tool type
        #[cfg(feature = "mcp-server")]
        let result = {
            let config = crate::config::get_global_config().await;
            match &config.tool {
                ToolSection::Mock(mock_config) => self.execute_mock(&params, mock_config).await?,
            }
        };

        #[cfg(not(feature = "mcp-server"))]
        let result = ToolResult::success(Some(serde_json::json!({
            "message": "Tool execution requires mcp-server feature"
        })));

        // Add token usage metadata
        let token_meta = json!({
            "total_prompt_tokens": token_usage.total_prompt_tokens,
            "total_completion_tokens": token_usage.total_completion_tokens,
            "total_tokens": token_usage.total_tokens,
            "model_identifier": token_usage.model_identifier,
        });

        let result_with_meta = result.with_metadata("token_usage", token_meta);

        Ok(result_with_meta)
    }
}

impl ImageGenerationTool {
    /// Execute mock image generation
    #[cfg(feature = "mcp-server")]
    async fn execute_mock(&self, _params: &ImageGenerationParams, mock_config: &crate::config::MockToolConfig) -> Result<ToolResult> {
        // Read the static image file
        let image_path = Path::new(&mock_config.static_image_path);

        if !image_path.exists() {
            return Err(BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason: format!("Image file not found: {}", image_path.display()),
            });
        }

        // Read image bytes
        let image_bytes = tokio::fs::read(image_path)
            .await
            .map_err(|e| BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason: format!("Failed to read image file {}: {}", image_path.display(), e),
            })?;

        // Detect MIME type
        let mime_type = mime_guess::from_path(image_path)
            .first()
            .unwrap_or(mime_guess::mime::IMAGE_PNG)
            .to_string();

        // Get actual image dimensions (would require image crate, using hardcoded for now)
        // In a real implementation, you'd use the `image` crate to decode and get dimensions
        let (width, height) = (500, 500); // Actual dimensions of image_1.png

        // Encode to base64
        let base64_data = base64::engine::general_purpose::STANDARD.encode(&image_bytes);

        // Create data URI
        let data_uri = format!("data:{};base64,{}", mime_type, base64_data);

        // Build response
        let image = GeneratedImage {
            data_uri,
            width,
            height,
            mime_type,
        };

        let response = ImageGenerationResponse {
            images: vec![image],
        };

        Ok(ToolResult::success_with(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        // Test that default functions return correct values
        assert_eq!(default_width(), Some(1024));
        assert_eq!(default_height(), Some(1024));
        assert_eq!(default_num_images(), Some(1));
    }

    #[test]
    fn test_token_usage_calculation() {
        let token_usage = TokenUsage {
            total_prompt_tokens: 100 * 4,
            total_completion_tokens: 0,
            total_tokens: 100 * 4,
            model_identifier: "mock/v1".to_string(),
        };

        assert_eq!(token_usage.total_prompt_tokens, 400);
        assert_eq!(token_usage.total_completion_tokens, 0);
        assert_eq!(token_usage.total_tokens, 400);
    }

    // Note: Integration tests requiring tokio are in the binary's test suite
    // since tokio is only available with the mcp-server feature
}
