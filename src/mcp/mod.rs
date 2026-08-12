//! MCP (Model Context Protocol) server implementation for image generation
//!
//! This module provides rmcp-compatible tools by wrapping the existing tool implementations.

pub mod handler;
pub use handler::BrowserServer;

use crate::tools::{self, Tool, ToolContext, ToolResult as InternalToolResult};
use http::header::AUTHORIZATION;
use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content, Meta},
    tool, tool_router,
};

fn build_meta_map(result: &InternalToolResult, prefix: &str) -> Option<Meta> {
    if result.metadata.is_empty() {
        return None;
    }

    let mut meta_obj = serde_json::Map::new();
    for (key, value) in &result.metadata {
        let meta_key = format!("{}{}", prefix, key);
        if let Ok(obj) = serde_json::to_value(value) {
            meta_obj.insert(meta_key, obj);
        }
    }
    Some(Meta(meta_obj))
}

fn bearer_token_from_request_context(
    context: &rmcp::service::RequestContext<rmcp::RoleServer>,
) -> Option<String> {
    let parts = context.extensions.get::<http::request::Parts>()?;
    let header = parts.headers.get(AUTHORIZATION)?.to_str().ok()?;
    header
        .strip_prefix("Bearer ")
        .or_else(|| header.strip_prefix("bearer "))
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned)
}

/// Convert internal ToolResult to MCP CallToolResult
fn convert_result(result: InternalToolResult, prefix: &str) -> Result<CallToolResult, McpError> {
    let meta_map = build_meta_map(&result, prefix);

    if result.success {
        let (text, structured_content) = if let Some(data) = result.data {
            let text = serde_json::to_string_pretty(&data).unwrap_or_else(|_| data.to_string());
            (text, Some(data))
        } else {
            ("Success".to_string(), None)
        };

        Ok(CallToolResult {
            content: vec![Content::text(text)],
            structured_content,
            is_error: Some(false),
            meta: meta_map,
        })
    } else {
        let error_msg = result.error.unwrap_or_else(|| "Unknown error".to_string());
        let mut error_payload = serde_json::Map::new();
        error_payload.insert("error".to_string(), serde_json::Value::String(error_msg));
        if let Some(data) = result.data {
            error_payload.insert("data".to_string(), data);
        }
        let error_payload = serde_json::Value::Object(error_payload);
        let text = serde_json::to_string_pretty(&error_payload)
            .unwrap_or_else(|_| "{\"error\":\"Failed to serialize tool error\"}".to_string());

        Ok(CallToolResult {
            content: vec![Content::text(text)],
            structured_content: Some(error_payload),
            is_error: Some(true),
            meta: meta_map,
        })
    }
}

/// Macro to register MCP tools by automatically generating wrapper functions
macro_rules! register_mcp_tools {
    ($($mcp_name:ident => $tool_type:ty, $description:expr $(, output_schema = $output_schema:expr)?);* $(;)?) => {
        #[tool_router]
        impl BrowserServer {
            $(
                #[tool(description = $description $(, output_schema = $output_schema)?)]
                async fn $mcp_name(
                    &self,
                    params: Parameters<<$tool_type as Tool>::Params>,
                    request_context: rmcp::service::RequestContext<rmcp::RoleServer>,
                ) -> Result<CallToolResult, McpError> {
                    let mut context = ToolContext::from_bearer_token(
                        bearer_token_from_request_context(&request_context),
                    );
                    let tool = <$tool_type>::default();
                    let result = match tool.execute_typed(params.0, &mut context).await {
                        Ok(result) => result,
                        Err(error) => InternalToolResult::failure(error.to_string()),
                    };

                    // Get metadata prefix from global config
                    let config = crate::config::get_global_config().await;
                    let prefix = config.metadata.prefix.as_str();

                    convert_result(result, prefix)
                }
            )*
        }
    };
}

// Register all MCP tools using the macro
register_mcp_tools! {
    generate_image => tools::image_generation::ImageGenerationTool,
        "Generate images from text prompts. Returns base64-encoded image data in the data_base64 field (no data URI prefix).",
        output_schema = rmcp::handler::server::tool::schema_for_type::<tools::image_generation::ImageGenerationResponse>();

    edit_image => tools::image_generation::ImageEditTool,
        "Edit images from text prompts plus an input image. Returns base64-encoded image data in the data_base64 field (no data URI prefix).",
        output_schema = rmcp::handler::server::tool::schema_for_type::<tools::image_generation::ImageGenerationResponse>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolResult as InternalToolResult;

    #[test]
    fn convert_result_preserves_structured_error_payload() {
        let result = InternalToolResult::failure_with_data(
            "blocked",
            serde_json::json!({
                "type": "content_filter",
                "error_description": "blocked by policy"
            }),
        );

        let converted = convert_result(result, "chat.erato/")
            .expect("conversion should return MCP tool result");
        assert_eq!(converted.is_error, Some(true));
        assert_eq!(
            converted.structured_content,
            Some(serde_json::json!({
                "error": "blocked",
                "data": {
                    "type": "content_filter",
                    "error_description": "blocked by policy"
                }
            }))
        );

        let text = converted.content[0]
            .as_text()
            .expect("error content should be text");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&text.text)
                .expect("error content should be JSON"),
            serde_json::json!({
                "error": "blocked",
                "data": {
                    "type": "content_filter",
                    "error_description": "blocked by policy"
                }
            })
        );
    }

    #[test]
    fn convert_result_wraps_plain_errors_as_json() {
        let converted = convert_result(InternalToolResult::failure("source image not found"), "")
            .expect("conversion should return MCP tool result");

        assert_eq!(converted.is_error, Some(true));
        assert_eq!(
            converted.structured_content,
            Some(serde_json::json!({"error": "source image not found"}))
        );
        let text = converted.content[0]
            .as_text()
            .expect("error content should be text");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&text.text)
                .expect("error content should be JSON"),
            serde_json::json!({"error": "source image not found"})
        );
    }
}
