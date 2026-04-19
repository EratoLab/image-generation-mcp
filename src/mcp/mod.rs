//! MCP (Model Context Protocol) server implementation for image generation
//!
//! This module provides rmcp-compatible tools by wrapping the existing tool implementations.

pub mod handler;
pub use handler::BrowserServer;

use crate::tools::{self, Tool, ToolContext, ToolResult as InternalToolResult};
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
        let (text, structured_content) = if let Some(data) = result.data {
            let text = serde_json::to_string_pretty(&data).unwrap_or_else(|_| data.to_string());
            (text, Some(data))
        } else {
            (error_msg, None)
        };

        Ok(CallToolResult {
            content: vec![Content::text(text)],
            structured_content,
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
                ) -> Result<CallToolResult, McpError> {
                    let mut context = ToolContext::new();
                    let tool = <$tool_type>::default();
                    let result = tool.execute_typed(params.0, &mut context)
                        .await
                        .map_err(|e| McpError::internal_error(e.to_string(), None))?;

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
                "type": "content_filter",
                "error_description": "blocked by policy"
            }))
        );
    }
}
