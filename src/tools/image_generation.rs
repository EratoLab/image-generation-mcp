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
use reqwest::Url;
#[cfg(feature = "mcp-server")]
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
#[cfg(feature = "mcp-server")]
use std::path::Path;
#[cfg(feature = "mcp-server")]
use std::str::FromStr;

/// Parameters for image editing
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImageEditParams {
    /// Source image as a URI/data URI.
    pub image_uri: Option<String>,

    /// Source image as raw base64 payload (no data URI prefix).
    #[schemars(extend(
        "contentEncoding" = "base64",
        "chat.erato/file_content_field" = true
    ))]
    pub image_data_base64: Option<String>,

    /// MIME type for the source image.
    #[serde(default)]
    pub image_mime_type: Option<String>,

    /// Editing prompt for the image.
    pub prompt: String,

    /// Optional width of the edited image (default: 1024)
    #[serde(default = "default_width")]
    pub width: Option<u32>,

    /// Optional height of the edited image (default: 1024)
    #[serde(default = "default_height")]
    pub height: Option<u32>,

    /// Number of edited images to generate (default: 1)
    #[serde(default = "default_num_images")]
    pub num_images: Option<u32>,
}

/// Edit-image tool
#[derive(Debug, Default)]
pub struct ImageEditTool;

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
        context: &mut ToolContext<'_>,
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
                    self.execute_genai(&params, genai_config, context).await?
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

#[async_trait]
impl Tool for ImageEditTool {
    type Params = ImageEditParams;

    fn name(&self) -> &str {
        "edit_image"
    }

    async fn execute_typed(
        &self,
        params: Self::Params,
        context: &mut ToolContext<'_>,
    ) -> Result<ToolResult> {
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
                    self.execute_genai(&params, genai_config, context).await?
                }
            }
        };

        #[cfg(not(feature = "mcp-server"))]
        let (result, token_usage) = (
            ToolResult::failure("Image editing is not available without mcp-server feature"),
            TokenUsage {
                total_prompt_tokens: 0,
                total_completion_tokens: 0,
                total_tokens: 0,
                model_identifier: "none".to_string(),
            },
        );

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
    fn is_gemini_provider(config: &crate::config::GenAiToolConfig) -> bool {
        config.provider_kind == "gemini"
    }

    #[cfg(feature = "mcp-server")]
    fn closest_aspect_ratio(width: u32, height: u32) -> &'static str {
        if width == 0 || height == 0 {
            return "1:1";
        }

        let target = width as f64 / height as f64;
        let candidates: [(&str, f64); 7] = [
            ("1:1", 1.0),
            ("16:9", 16.0 / 9.0),
            ("9:16", 9.0 / 16.0),
            ("4:3", 4.0 / 3.0),
            ("3:4", 3.0 / 4.0),
            ("3:2", 3.0 / 2.0),
            ("2:3", 2.0 / 3.0),
        ];

        let mut best = candidates[0];
        let mut best_delta = (target - best.1).abs();
        for candidate in &candidates[1..] {
            let delta = (target - candidate.1).abs();
            if delta < best_delta {
                best = *candidate;
                best_delta = delta;
            }
        }

        best.0
    }

    #[cfg(feature = "mcp-server")]
    fn map_usage_from_vertex(payload: &Value, model_identifier: String) -> TokenUsage {
        let usage = payload.get("usageMetadata");
        TokenUsage {
            total_prompt_tokens: usage
                .and_then(|u| u.get("promptTokenCount"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
            total_completion_tokens: usage
                .and_then(|u| u.get("candidatesTokenCount"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
            total_tokens: usage
                .and_then(|u| u.get("totalTokenCount"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
            model_identifier,
        }
    }

    #[cfg(feature = "mcp-server")]
    fn default_vertex_base_url(config: &crate::config::GenAiToolConfig) -> String {
        config
            .region
            .as_deref()
            .map(str::trim)
            .filter(|region| !region.is_empty())
            .map(|region| {
                crate::config::GenAiToolConfig::VERTEX_AI_REGION_BASE_URL_TEMPLATE
                    .replace("{region}", region)
            })
            .unwrap_or_else(|| crate::config::GenAiToolConfig::VERTEX_AI_BASE_URL.to_string())
    }

    #[cfg(feature = "mcp-server")]
    fn build_custom_http_client(
        config: &crate::config::GenAiToolConfig,
        context: Option<&ToolContext<'_>>,
    ) -> Result<reqwest::Client> {
        let request_headers = config.additional_request_headers_map();

        let mut header_map = HeaderMap::new();
        for (key, value) in request_headers {
            let rendered_value = context
                .map(|context| context.render_header_template(&value))
                .unwrap_or(value);
            let name = HeaderName::from_str(&key).map_err(|e| {
                BrowserError::ConfigError(format!("Invalid header name '{}': {}", key, e))
            })?;
            let value = HeaderValue::from_str(&rendered_value).map_err(|e| {
                BrowserError::ConfigError(format!("Invalid header value for '{}': {}", key, e))
            })?;
            header_map.insert(name, value);
        }

        reqwest::ClientBuilder::new()
            .default_headers(header_map)
            .build()
            .map_err(|e| BrowserError::ConfigError(format!("Failed to build HTTP client: {}", e)))
    }

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
        context: &ToolContext<'_>,
    ) -> Result<(ToolResult, TokenUsage)> {
        if Self::is_gemini_provider(genai_config) {
            return self
                .execute_vertex_gemini(params, genai_config, context)
                .await;
        }

        let client = Self::build_genai_client(genai_config, context)?;

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
    async fn execute_vertex_gemini(
        &self,
        params: &ImageGenerationParams,
        genai_config: &crate::config::GenAiToolConfig,
        context: &ToolContext<'_>,
    ) -> Result<(ToolResult, TokenUsage)> {
        let width = params.width.unwrap_or(1024);
        let height = params.height.unwrap_or(1024);
        let num_images = params.num_images.unwrap_or(1);
        if num_images > i32::MAX as u32 {
            return Err(BrowserError::InvalidArgument(
                "num_images is too large".to_string(),
            ));
        }
        if width == 0 || height == 0 {
            return Err(BrowserError::InvalidArgument(
                "width and height must be greater than zero".to_string(),
            ));
        }

        let response_format = genai_config
            .response_format
            .as_deref()
            .unwrap_or("b64_json");
        if response_format != "b64_json" {
            return Err(BrowserError::ConfigError(format!(
                "Gemini/Vertex image generation only supports response_format='b64_json', got '{}'",
                response_format
            )));
        }

        let base_url = genai_config
            .base_url
            .clone()
            .unwrap_or_else(|| Self::default_vertex_base_url(genai_config));
        let base_url_for_err = base_url.clone();
        let normalized_base = if base_url.ends_with('/') {
            base_url
        } else {
            format!("{}/", base_url)
        };

        let mut request_url = Url::parse(&format!(
            "{}models/{}:generateContent",
            normalized_base, genai_config.model_name
        ))
        .map_err(|e| {
            BrowserError::ConfigError(format!(
                "Invalid Gemini/Vertex base_url '{}': {}",
                base_url_for_err, e
            ))
        })?;

        {
            let mut query = request_url.query_pairs_mut();
            for (key, value) in genai_config.additional_request_parameters_map() {
                query.append_pair(&key, &value);
            }
            if let Some(api_key) = &genai_config.api_key {
                query.append_pair("key", api_key);
            }
        }

        let aspect_ratio = ImageGenerationTool::closest_aspect_ratio(width, height);
        let request_body = json!({
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": params.prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "candidateCount": num_images,
                "imageConfig": {
                    "aspectRatio": aspect_ratio
                }
            }
        });

        let client = Self::build_custom_http_client(genai_config, Some(context))?;
        let response = client
            .post(request_url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason: format!("Gemini/Vertex image generation request failed: {}", e),
            })?;

        let status = response.status();
        let body_text = response
            .text()
            .await
            .map_err(|e| BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason: format!("Failed to read Gemini/Vertex response body: {}", e),
            })?;

        if !status.is_success() {
            if let Some(content_filter_error) =
                Self::parse_content_filter_error_from_body(&body_text)
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

            let reason = if body_text.is_empty() {
                format!("Gemini/Vertex API returned HTTP {}", status)
            } else if body_text
                .to_ascii_lowercase()
                .contains("multi-modal output is not supported")
            {
                format!(
                    "Gemini/Vertex API returned HTTP {}: model '{}' does not support IMAGE output. Use an image-capable model such as 'gemini-2.5-flash-image'. Raw response: {}",
                    status, genai_config.model_name, body_text
                )
            } else {
                format!("Gemini/Vertex API returned HTTP {}: {}", status, body_text)
            };
            return Err(BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason,
            });
        }

        let payload: Value =
            serde_json::from_str(&body_text).map_err(|e| BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason: format!("Failed to parse Gemini/Vertex response JSON: {}", e),
            })?;

        let mut images = Vec::new();
        if let Some(candidates) = payload.get("candidates").and_then(Value::as_array) {
            for candidate in candidates {
                let Some(parts) = candidate
                    .get("content")
                    .and_then(|content| content.get("parts"))
                    .and_then(Value::as_array)
                else {
                    continue;
                };

                for part in parts {
                    let Some(inline_data) = part.get("inlineData").and_then(Value::as_object)
                    else {
                        continue;
                    };

                    let Some(data_base64) = inline_data.get("data").and_then(Value::as_str) else {
                        continue;
                    };
                    let mime_type = inline_data
                        .get("mimeType")
                        .and_then(Value::as_str)
                        .unwrap_or("image/png")
                        .to_string();

                    images.push(GeneratedImage {
                        data_base64: data_base64.to_string(),
                        width,
                        height,
                        mime_type,
                    });
                }
            }
        }

        if images.is_empty() {
            return Err(BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason: format!(
                    "Gemini/Vertex returned no inline image data. Response payload: {}",
                    payload
                ),
            });
        }

        let model_identifier = payload
            .get("modelVersion")
            .and_then(Value::as_str)
            .unwrap_or(&genai_config.model_name)
            .to_string();
        let token_usage = Self::map_usage_from_vertex(&payload, model_identifier);

        let response = ImageGenerationResponse { images };
        Ok((ToolResult::success_with(response), token_usage))
    }

    #[cfg(feature = "mcp-server")]
    fn build_genai_client(
        config: &crate::config::GenAiToolConfig,
        context: &ToolContext<'_>,
    ) -> Result<GenaiClient> {
        let base_url = config.base_url.clone();
        let request_params = config.additional_request_parameters_map();
        let custom_client = Self::build_custom_http_client(config, Some(context))?;

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

#[cfg(feature = "mcp-server")]
impl ImageEditTool {
    fn is_gemini_provider(config: &crate::config::GenAiToolConfig) -> bool {
        config.provider_kind == "gemini" || config.provider_kind == "vertex_ai"
    }

    fn strip_data_uri_prefix(data: &str) -> String {
        match data.find("base64,") {
            Some(idx) => data[(idx + "base64,".len())..].to_string(),
            None => data.to_string(),
        }
    }

    fn sanitize_image_mime_type(params: &ImageEditParams) -> String {
        params
            .image_mime_type
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "image/png".to_string())
            .trim()
            .to_string()
    }

    fn image_output_mime(genai_config: &crate::config::GenAiToolConfig) -> String {
        let configured = genai_config
            .response_format
            .as_deref()
            .unwrap_or("image/png")
            .trim()
            .to_ascii_lowercase();

        if configured == "jpeg" || configured == "jpg" {
            "image/jpeg".to_string()
        } else if configured == "webp" {
            "image/webp".to_string()
        } else if configured == "image/jpeg"
            || configured == "image/webp"
            || configured == "image/png"
        {
            configured
        } else {
            "image/png".to_string()
        }
    }

    fn image_input_part(params: &ImageEditParams) -> Result<Value> {
        let image_mime = Self::sanitize_image_mime_type(params);
        if let Some(uri) = params
            .image_uri
            .as_ref()
            .filter(|uri| !uri.trim().is_empty())
        {
            return Ok(json!({
                "file_data": {
                    "mime_type": image_mime,
                    "file_uri": uri.trim()
                }
            }));
        }

        if let Some(image_data_base64) = params
            .image_data_base64
            .as_ref()
            .filter(|data| !data.trim().is_empty())
        {
            return Ok(json!({
                "inline_data": {
                    "mime_type": image_mime,
                    "data": Self::strip_data_uri_prefix(image_data_base64.trim())
                }
            }));
        }

        Err(BrowserError::InvalidArgument(
            "Either image_uri or image_data_base64 must be provided".to_string(),
        ))
    }

    async fn execute_mock(
        &self,
        _params: &ImageEditParams,
        mock_config: &crate::config::MockToolConfig,
    ) -> Result<ToolResult> {
        let generation_params = ImageGenerationParams {
            prompt: "image edit request".to_string(),
            width: default_width(),
            height: default_height(),
            num_images: default_num_images(),
        };

        let tool = ImageGenerationTool;
        tool.execute_mock(&generation_params, mock_config).await
    }

    async fn execute_genai(
        &self,
        params: &ImageEditParams,
        genai_config: &crate::config::GenAiToolConfig,
        context: &ToolContext<'_>,
    ) -> Result<(ToolResult, TokenUsage)> {
        if Self::is_gemini_provider(genai_config) {
            return self
                .execute_vertex_gemini(params, genai_config, context)
                .await;
        }

        let reason = format!(
            "Image editing is not yet implemented for provider '{}'; supported providers: gemini/vertex_ai (via generateContent).",
            genai_config.provider_kind
        );
        let token_usage = TokenUsage {
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            total_tokens: 0,
            model_identifier: genai_config.model_name.clone(),
        };

        Ok((ToolResult::failure(reason), token_usage))
    }

    async fn execute_vertex_gemini(
        &self,
        params: &ImageEditParams,
        genai_config: &crate::config::GenAiToolConfig,
        context: &ToolContext<'_>,
    ) -> Result<(ToolResult, TokenUsage)> {
        let width = params.width.unwrap_or(1024);
        let height = params.height.unwrap_or(1024);
        let num_images = params.num_images.unwrap_or(1);

        if params.prompt.trim().is_empty() {
            return Err(BrowserError::InvalidArgument(
                "prompt is required for image editing".to_string(),
            ));
        }

        if width == 0 || height == 0 {
            return Err(BrowserError::InvalidArgument(
                "width and height must be greater than zero".to_string(),
            ));
        }

        if num_images == 0 {
            return Err(BrowserError::InvalidArgument(
                "num_images must be at least 1".to_string(),
            ));
        }

        if num_images > i32::MAX as u32 {
            return Err(BrowserError::InvalidArgument(
                "num_images is too large".to_string(),
            ));
        }

        let image_input = Self::image_input_part(params)?;
        let aspect_ratio = ImageGenerationTool::closest_aspect_ratio(width, height);
        let response_format = Self::image_output_mime(genai_config);

        let base_url = genai_config
            .base_url
            .clone()
            .unwrap_or_else(|| ImageGenerationTool::default_vertex_base_url(genai_config));
        let base_url_for_err = base_url.clone();
        let normalized_base = if base_url.ends_with('/') {
            base_url
        } else {
            format!("{}/", base_url)
        };

        let mut request_url = Url::parse(&format!(
            "{}models/{}:generateContent",
            normalized_base, genai_config.model_name
        ))
        .map_err(|e| {
            BrowserError::ConfigError(format!(
                "Invalid Gemini/Vertex base_url '{}': {}",
                base_url_for_err, e
            ))
        })?;

        {
            let mut query = request_url.query_pairs_mut();
            for (key, value) in genai_config.additional_request_parameters_map() {
                query.append_pair(&key, &value);
            }
            if let Some(api_key) = &genai_config.api_key {
                query.append_pair("key", api_key);
            }
        }

        let request_body = json!({
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        image_input,
                        { "text": params.prompt.as_str() },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "candidateCount": num_images,
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageOutputOptions": {
                        "mimeType": response_format
                    }
                }
            }
        });

        let client = ImageGenerationTool::build_custom_http_client(genai_config, Some(context))?;
        let response = client
            .post(request_url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason: format!("Gemini/Vertex image edit request failed: {}", e),
            })?;

        let status = response.status();
        let body_text = response
            .text()
            .await
            .map_err(|e| BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason: format!(
                    "Failed to read Gemini/Vertex image edit response body: {}",
                    e
                ),
            })?;

        if !status.is_success() {
            if let Some(content_filter_error) =
                ImageGenerationTool::parse_content_filter_error_from_body(&body_text)
            {
                let result = ToolResult::failure_with_data(
                    "Image edit request blocked by content filter.",
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

            let reason = if body_text.is_empty() {
                format!("Gemini/Vertex API returned HTTP {}", status)
            } else {
                format!("Gemini/Vertex API returned HTTP {}: {}", status, body_text)
            };
            return Err(BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason,
            });
        }

        let payload: Value =
            serde_json::from_str(&body_text).map_err(|e| BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason: format!(
                    "Failed to parse Gemini/Vertex image edit response JSON: {}",
                    e
                ),
            })?;

        let mut images = Vec::new();
        if let Some(candidates) = payload.get("candidates").and_then(Value::as_array) {
            for candidate in candidates {
                let Some(parts) = candidate
                    .get("content")
                    .and_then(|content| content.get("parts"))
                    .and_then(Value::as_array)
                else {
                    continue;
                };

                for part in parts {
                    let Some(inline_data) = part.get("inlineData").and_then(Value::as_object)
                    else {
                        continue;
                    };

                    let Some(data_base64) = inline_data.get("data").and_then(Value::as_str) else {
                        continue;
                    };
                    let mime_type = inline_data
                        .get("mimeType")
                        .and_then(Value::as_str)
                        .unwrap_or("image/png")
                        .to_string();

                    images.push(GeneratedImage {
                        data_base64: data_base64.to_string(),
                        width,
                        height,
                        mime_type,
                    });
                }
            }
        }

        if images.is_empty() {
            return Err(BrowserError::ToolExecutionFailed {
                tool: self.name().to_string(),
                reason: format!(
                    "Gemini/Vertex image edit returned no inline image data. Response payload: {}",
                    payload
                ),
            });
        }

        let model_identifier = payload
            .get("modelVersion")
            .and_then(Value::as_str)
            .unwrap_or(&genai_config.model_name)
            .to_string();
        let token_usage = ImageGenerationTool::map_usage_from_vertex(&payload, model_identifier);

        let response = ImageGenerationResponse { images };
        Ok((ToolResult::success_with(response), token_usage))
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

    #[test]
    fn test_closest_aspect_ratio_mapping() {
        assert_eq!(ImageGenerationTool::closest_aspect_ratio(1024, 1024), "1:1");
        assert_eq!(
            ImageGenerationTool::closest_aspect_ratio(1920, 1080),
            "16:9"
        );
        assert_eq!(
            ImageGenerationTool::closest_aspect_ratio(1080, 1920),
            "9:16"
        );
    }

    #[test]
    fn test_map_usage_from_vertex_payload() {
        let payload = json!({
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 7,
                "totalTokenCount": 18
            }
        });
        let usage = ImageGenerationTool::map_usage_from_vertex(&payload, "gemini-x".to_string());
        assert_eq!(usage.total_prompt_tokens, 11);
        assert_eq!(usage.total_completion_tokens, 7);
        assert_eq!(usage.total_tokens, 18);
        assert_eq!(usage.model_identifier, "gemini-x");
    }

    // Note: Integration tests requiring tokio are in the binary's test suite
    // since tokio is only available with the mcp-server feature
}
