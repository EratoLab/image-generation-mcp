//! Image generation tool with mock implementation
//!
//! This module provides a mock image generation tool that returns a static image.
//! Future versions will support multiple backends (OpenAI, Stability AI, etc.)

use crate::config::ToolSection;
use crate::error::Result;
use crate::tools::{Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
#[cfg(feature = "mcp-server")]
use serde_json::Value;
use serde_json::json;

#[cfg(feature = "mcp-server")]
use crate::error::BrowserError;

#[cfg(feature = "mcp-server")]
use base64::Engine;

#[cfg(feature = "mcp-server")]
use genai::adapter::AdapterKind;
#[cfg(feature = "mcp-server")]
use genai::chat::{BinarySource, ContentPart, ImageRequest};
#[cfg(feature = "mcp-server")]
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
#[cfg(feature = "mcp-server")]
use genai::webc;
#[cfg(feature = "mcp-server")]
use genai::{Client as GenaiClient, ModelIden, ServiceTarget};
#[cfg(feature = "mcp-server")]
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
#[cfg(feature = "mcp-server")]
use std::path::Path;
#[cfg(feature = "mcp-server")]
use std::str::FromStr;

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
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GeneratedImage {
    #[schemars(extend(
        "contentEncoding" = "base64",
        "chat.erato/file_content_field" = true
    ))]
    pub data_base64: String,
    pub width: u32,
    pub height: u32,
    pub mime_type: String,
}

/// Response wrapper for generated images
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImageGenerationResponse {
    pub images: Vec<GeneratedImage>,
}

#[async_trait]
impl Tool for ImageGenerationTool {
    type Params = ImageGenerationParams;

    fn name(&self) -> &str {
        "generate_image"
    }

    async fn execute_typed(
        &self,
        params: Self::Params,
        _context: &mut ToolContext<'_>,
    ) -> Result<ToolResult> {
        // Execute based on tool type
        #[cfg(feature = "mcp-server")]
        let (result, token_usage) = {
            let config = crate::config::get_global_config().await;
            match &config.tool {
                ToolSection::Mock(mock_config) => {
                    let result = self.execute_mock(&params, mock_config).await?;
                    let token_usage = TokenUsage {
                        total_prompt_tokens: (params.prompt.len() as u64) * 4,
                        total_completion_tokens: 0,
                        total_tokens: (params.prompt.len() as u64) * 4,
                        model_identifier: "mock/v1".to_string(),
                    };
                    (result, token_usage)
                }
                ToolSection::GenAi(genai_config) => {
                    self.execute_genai(&params, genai_config).await?
                }
            }
        };

        #[cfg(not(feature = "mcp-server"))]
        let (result, token_usage) = (
            ToolResult::success(Some(serde_json::json!({
                "message": "Tool execution requires mcp-server feature"
            }))),
            TokenUsage {
                total_prompt_tokens: 0,
                total_completion_tokens: 0,
                total_tokens: 0,
                model_identifier: "none".to_string(),
            },
        );

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
    #[cfg(feature = "mcp-server")]
    fn message_indicates_content_filter(message: &str) -> bool {
        let message = message.to_ascii_lowercase();
        message.contains("content rejected")
            || message.contains("content filter")
            || message.contains("content policy")
            || message.contains("sexual content detection")
            || message.contains("violence detection")
            || message.contains("hate detection")
            || message.contains("self-harm detection")
            || message.contains("self harm detection")
    }

    #[cfg(feature = "mcp-server")]
    fn inferred_filter_details_from_message(message: &str) -> Option<Value> {
        let msg = message.to_ascii_lowercase();
        let mut details = serde_json::Map::new();

        if msg.contains("sexual") {
            details.insert(
                "sexual".to_string(),
                json!({"filtered": true, "severity": "medium"}),
            );
        }
        if msg.contains("violence") {
            details.insert(
                "violence".to_string(),
                json!({"filtered": true, "severity": "medium"}),
            );
        }
        if msg.contains("hate") {
            details.insert(
                "hate".to_string(),
                json!({"filtered": true, "severity": "medium"}),
            );
        }
        if msg.contains("self-harm") || msg.contains("self harm") {
            details.insert(
                "self_harm".to_string(),
                json!({"filtered": true, "severity": "medium"}),
            );
        }

        if details.is_empty() {
            None
        } else {
            Some(Value::Object(details))
        }
    }

    #[cfg(feature = "mcp-server")]
    fn parse_content_filter_error_payload(payload: &Value) -> Option<Value> {
        let mut candidates = vec![payload];
        if let Some(error_obj) = payload.get("error") {
            candidates.push(error_obj);
        }

        for candidate in candidates {
            let error_type = candidate.get("type").and_then(Value::as_str);
            let error_code = candidate.get("code").and_then(Value::as_str);
            let message = candidate
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| payload.get("message").and_then(Value::as_str))
                .unwrap_or_default();

            let is_content_filter = error_type == Some("content_filter")
                || error_code == Some("content_filter")
                || Self::message_indicates_content_filter(message);

            if !is_content_filter {
                continue;
            }

            let error_description = candidate
                .get("error_description")
                .and_then(Value::as_str)
                .or(Some(message))
                .unwrap_or("The request was filtered by content policy.")
                .to_string();

            let filter_details = candidate
                .get("filter_details")
                .cloned()
                .or_else(|| candidate.get("content_filter_result").cloned())
                .or_else(|| {
                    candidate
                        .get("innererror")
                        .and_then(|inner| inner.get("content_filter_result"))
                        .cloned()
                })
                .or_else(|| Self::inferred_filter_details_from_message(&error_description));

            return Some(json!({
                "type": "content_filter",
                "error_description": error_description,
                "filter_details": filter_details
            }));
        }

        None
    }

    #[cfg(feature = "mcp-server")]
    fn parse_content_filter_error_from_body(body: &str) -> Option<Value> {
        serde_json::from_str::<Value>(body)
            .ok()
            .and_then(|payload| Self::parse_content_filter_error_payload(&payload))
    }

    #[cfg(feature = "mcp-server")]
    fn parse_content_filter_error_from_genai_error(error: &genai::Error) -> Option<Value> {
        match error {
            genai::Error::ChatResponse { body, .. } => {
                Self::parse_content_filter_error_payload(body)
            }
            genai::Error::WebModelCall { webc_error, .. }
            | genai::Error::WebAdapterCall { webc_error, .. } => match webc_error {
                webc::Error::ResponseFailedStatus { body, .. }
                | webc::Error::ResponseFailedNotJson { body, .. } => {
                    Self::parse_content_filter_error_from_body(body)
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Execute mock image generation
    #[cfg(feature = "mcp-server")]
    async fn execute_mock(
        &self,
        _params: &ImageGenerationParams,
        mock_config: &crate::config::MockToolConfig,
    ) -> Result<ToolResult> {
        // Read the static image file
        let image_path = Path::new(&mock_config.static_image_path);

        if !image_path.exists() {
            return Err(BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason: format!("Image file not found: {}", image_path.display()),
            });
        }

        // Read image bytes
        let image_bytes =
            tokio::fs::read(image_path)
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

        // Build response
        let image = GeneratedImage {
            data_base64: base64_data,
            width,
            height,
            mime_type,
        };

        let response = ImageGenerationResponse {
            images: vec![image],
        };

        Ok(ToolResult::success_with(response))
    }

    #[cfg(feature = "mcp-server")]
    async fn execute_genai(
        &self,
        params: &ImageGenerationParams,
        genai_config: &crate::config::GenAiToolConfig,
    ) -> Result<(ToolResult, TokenUsage)> {
        let client = Self::build_genai_client(genai_config)?;

        let width = params.width.unwrap_or(1024);
        let height = params.height.unwrap_or(1024);
        let num_images = params.num_images.unwrap_or(1);
        if num_images > i32::MAX as u32 {
            return Err(BrowserError::InvalidArgument(
                "num_images is too large".to_string(),
            ));
        }

        let mut image_request = ImageRequest::from_prompt(&params.prompt)
            .with_n(num_images as i32)
            .with_size(format!("{}x{}", width, height));

        if let Some(quality) = &genai_config.image_quality {
            image_request = image_request.with_quality(quality.clone());
        }
        if let Some(style) = &genai_config.image_style {
            image_request = image_request.with_style(style.clone());
        }
        let response_format = genai_config
            .response_format
            .clone()
            .unwrap_or_else(|| "b64_json".to_string());
        image_request = image_request.with_response_format(response_format);

        let image_response = match client
            .exec_image_generation(&genai_config.model_name, image_request, None)
            .await
        {
            Ok(response) => response,
            Err(err) => {
                if let Some(content_filter_error) =
                    Self::parse_content_filter_error_from_genai_error(&err)
                {
                    let result = ToolResult::failure_with_data(
                        "Generation request blocked by content filter.",
                        content_filter_error,
                    );
                    let token_usage = TokenUsage {
                        total_prompt_tokens: 0,
                        total_completion_tokens: 0,
                        total_tokens: 0,
                        model_identifier: genai_config.model_name.clone(),
                    };
                    return Ok((result, token_usage));
                }

                return Err(BrowserError::ToolExecutionFailed {
                    tool: self.name().to_string(),
                    reason: format!("GenAI image generation failed: {}", err),
                });
            }
        };

        let mut images = Vec::new();
        for part in image_response.images {
            let ContentPart::Binary(binary) = part else {
                return Err(BrowserError::ToolExecutionFailed {
                    tool: self.name().to_string(),
                    reason: "Unexpected non-binary image content in provider response".to_string(),
                });
            };

            let data_base64 = match binary.source {
                BinarySource::Base64(base64_data) => base64_data.to_string(),
                BinarySource::Url(_) => {
                    return Err(BrowserError::ToolExecutionFailed {
                        tool: self.name().to_string(),
                        reason: "Provider returned image URL; expected base64 content".to_string(),
                    });
                }
            };

            images.push(GeneratedImage {
                data_base64,
                width,
                height,
                mime_type: binary.content_type,
            });
        }

        if images.is_empty() {
            return Err(BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason: "No images were generated".to_string(),
            });
        }

        let usage = image_response.usage.clone();
        let token_usage = TokenUsage {
            total_prompt_tokens: usage
                .as_ref()
                .and_then(|u| u.prompt_tokens)
                .unwrap_or(0)
                .max(0) as u64,
            total_completion_tokens: usage
                .as_ref()
                .and_then(|u| u.completion_tokens)
                .unwrap_or(0)
                .max(0) as u64,
            total_tokens: usage
                .as_ref()
                .and_then(|u| u.total_tokens)
                .unwrap_or(0)
                .max(0) as u64,
            model_identifier: image_response.model_iden.to_string(),
        };

        let response = ImageGenerationResponse { images };
        Ok((ToolResult::success_with(response), token_usage))
    }

    #[cfg(feature = "mcp-server")]
    fn build_genai_client(config: &crate::config::GenAiToolConfig) -> Result<GenaiClient> {
        let base_url = config.base_url.clone();
        let request_params = config.additional_request_parameters_map();
        let request_headers = config.additional_request_headers_map();

        let mut header_map = HeaderMap::new();
        for (key, value) in request_headers {
            let name = HeaderName::from_str(&key).map_err(|e| {
                BrowserError::ConfigError(format!("Invalid header name '{}': {}", key, e))
            })?;
            let value = HeaderValue::from_str(&value).map_err(|e| {
                BrowserError::ConfigError(format!("Invalid header value for '{}': {}", key, e))
            })?;
            header_map.insert(name, value);
        }

        let custom_client = reqwest::ClientBuilder::new()
            .default_headers(header_map)
            .build()
            .map_err(|e| {
                BrowserError::ConfigError(format!("Failed to build HTTP client: {}", e))
            })?;

        let provider_kind = config.provider_kind.clone();
        let model_name = config.model_name.clone();
        let api_key = config.api_key.clone();

        let genai_client = genai::ClientBuilder::default()
            .with_reqwest(custom_client)
            .with_service_target_resolver(ServiceTargetResolver::from_resolver_fn(
                move |_service_target: ServiceTarget| -> std::result::Result<ServiceTarget, genai::resolver::Error> {
                    let adapter_kind = match provider_kind.as_str() {
                        "ollama" => AdapterKind::Ollama,
                        "openai" => AdapterKind::OpenAI,
                        _ => {
                            return Err(genai::resolver::Error::Custom(
                                "Unknown provider kind".to_string(),
                            ));
                        }
                    };

                    let mut endpoint = Self::default_endpoint(adapter_kind);
                    if let Some(base_url) = base_url.clone() {
                        let mut url_str = base_url;
                        if !request_params.is_empty() {
                            let mut first_param = true;
                            for (key, value) in &request_params {
                                if first_param {
                                    if url_str.contains('?') {
                                        url_str.push('&');
                                    } else {
                                        url_str.push('?');
                                    }
                                    first_param = false;
                                } else {
                                    url_str.push('&');
                                }
                                url_str.push_str(&format!("{}={}", key, value));
                            }
                        }
                        endpoint = Endpoint::from_owned(url_str);
                    }

                    let auth = if let Some(api_key) = api_key.clone() {
                        AuthData::from_single(api_key)
                    } else {
                        AuthData::from_single("PLACEHOLDER")
                    };

                    let model = ModelIden::new(adapter_kind, model_name.clone());
                    Ok(ServiceTarget {
                        endpoint,
                        auth,
                        model,
                    })
                },
            ))
            .build();

        Ok(genai_client)
    }

    #[cfg(feature = "mcp-server")]
    fn default_endpoint(kind: AdapterKind) -> Endpoint {
        match kind {
            AdapterKind::OpenAI => Endpoint::from_static("https://api.openai.com/v1/"),
            AdapterKind::Ollama => Endpoint::from_static("http://localhost:11434/v1/"),
            _ => Endpoint::from_static("https://api.openai.com/v1/"),
        }
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

    #[test]
    fn test_parse_content_filter_error_payload_infers_from_bad_request_message() {
        let payload = json!({
            "error": {
                "code": "Bad Request",
                "message": "Content rejected due to sexual content detection in the generated image.",
                "status": 400
            }
        });

        let parsed = ImageGenerationTool::parse_content_filter_error_payload(&payload)
            .expect("expected content filter payload");
        assert_eq!(parsed["type"], "content_filter");
        assert_eq!(
            parsed["error_description"],
            "Content rejected due to sexual content detection in the generated image."
        );
        assert_eq!(parsed["filter_details"]["sexual"]["filtered"], true);
    }

    #[test]
    fn test_parse_content_filter_error_payload_infers_violence_from_message() {
        let payload = json!({
            "error": {
                "code": "Bad Request",
                "message": "Content rejected due to violence detection in prompt.",
                "status": 400
            }
        });

        let parsed = ImageGenerationTool::parse_content_filter_error_payload(&payload)
            .expect("expected content filter payload");
        assert_eq!(parsed["type"], "content_filter");
        assert_eq!(parsed["filter_details"]["violence"]["filtered"], true);
    }

    // Note: Integration tests requiring tokio are in the binary's test suite
    // since tokio is only available with the mcp-server feature
}
