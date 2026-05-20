//! Image generation tools module
//!
//! This module provides tools for image generation operations.

pub mod image_generation;

// Re-export Params types for use by MCP layer
pub use image_generation::ImageEditParams;
pub use image_generation::ImageGenerationParams;

use crate::error::Result;
use base64::Engine;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Tool execution context
pub struct ToolContext<'a> {
    id_token_claims: Option<Value>,

    /// Phantom data to maintain lifetime
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> ToolContext<'a> {
    /// Create a new tool context
    pub fn new() -> Self {
        Self {
            id_token_claims: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a tool context from an inbound Authorization bearer token.
    pub fn from_bearer_token(bearer_token: Option<String>) -> Self {
        Self {
            id_token_claims: bearer_token
                .as_deref()
                .and_then(decode_jwt_claims_unverified),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Render supported header template variables against the inbound ID token.
    ///
    /// Supported variables follow Erato's chat-provider header context shape:
    /// `{{id_token.claims.<claim_name>}}`.
    pub fn render_header_template(&self, template: &str) -> String {
        let mut rendered = String::with_capacity(template.len());
        let mut remaining = template;

        while let Some(start) = remaining.find("{{") {
            let (prefix, after_start) = remaining.split_at(start);
            rendered.push_str(prefix);

            let after_start = &after_start[2..];
            let Some(end) = after_start.find("}}") else {
                rendered.push_str("{{");
                rendered.push_str(after_start);
                return rendered;
            };

            let expression = after_start[..end].trim();
            if let Some(value) = self.resolve_template_expression(expression) {
                rendered.push_str(&value);
            } else if expression.starts_with("id_token.claims.") {
                // Match Handlebars non-strict behavior for missing claims.
            } else {
                rendered.push_str("{{");
                rendered.push_str(&after_start[..end]);
                rendered.push_str("}}");
            }

            remaining = &after_start[(end + 2)..];
        }

        rendered.push_str(remaining);
        rendered
    }

    fn resolve_template_expression(&self, expression: &str) -> Option<String> {
        let path = expression.strip_prefix("id_token.claims.")?;
        let claims = self.id_token_claims.as_ref()?;
        claim_path_value(claims, path).map(json_value_to_header_value)
    }
}

impl<'a> Default for ToolContext<'a> {
    fn default() -> Self {
        Self::new()
    }
}

fn decode_jwt_claims_unverified(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let mut padded_payload = payload.to_string();
    while padded_payload.len() % 4 != 0 {
        padded_payload.push('=');
    }

    let bytes = base64::engine::general_purpose::URL_SAFE
        .decode(padded_payload)
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn claim_path_value<'a>(claims: &'a Value, path: &str) -> Option<&'a Value> {
    if let Some(path) = path.strip_prefix('[') {
        let end = path.find(']')?;
        let key = path[..end].trim_matches(|c| c == '"' || c == '\'');
        let value = claims.get(key)?;
        let rest = &path[(end + 1)..];
        return match rest.strip_prefix('.') {
            Some(rest) => dot_path_value(value, rest),
            None if rest.is_empty() => Some(value),
            None => None,
        };
    }

    dot_path_value(claims, path)
}

fn dot_path_value<'a>(claims: &'a Value, path: &str) -> Option<&'a Value> {
    let mut value = claims;
    for segment in path.split('.') {
        if segment.is_empty() {
            return None;
        }
        value = value.get(segment)?;
    }
    Some(value)
}

fn json_value_to_header_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
        Value::Null => String::new(),
    }
}

#[cfg(test)]
mod tool_context_tests {
    use super::*;

    fn jwt_with_claims(claims: Value) -> String {
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(claims.to_string());
        format!("e30.{payload}.sig")
    }

    #[test]
    fn renders_id_token_claim_from_bearer_token() {
        let token = jwt_with_claims(serde_json::json!({
            "oid": "user-123",
            "email": "alice@example.com"
        }));
        let context = ToolContext::from_bearer_token(Some(token));

        assert_eq!(
            context.render_header_template("X-User={{id_token.claims.oid}}"),
            "X-User=user-123"
        );
        assert_eq!(
            context.render_header_template("{{id_token.claims.email}}"),
            "alice@example.com"
        );
    }

    #[test]
    fn missing_id_token_claim_renders_empty() {
        let token = jwt_with_claims(serde_json::json!({"sub": "subject-1"}));
        let context = ToolContext::from_bearer_token(Some(token));

        assert_eq!(
            context.render_header_template("X-Missing={{id_token.claims.oid}}"),
            "X-Missing="
        );
    }

    #[test]
    fn renders_bracketed_claim_name() {
        let token = jwt_with_claims(serde_json::json!({
            "https://example.com/user_id": "external-user-1"
        }));
        let context = ToolContext::from_bearer_token(Some(token));

        assert_eq!(
            context.render_header_template("{{id_token.claims.[https://example.com/user_id]}}"),
            "external-user-1"
        );
    }
}

/// Result of tool execution
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolResult {
    /// Whether the tool execution was successful
    pub success: bool,

    /// Result data (JSON value)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,

    /// Error message if execution failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// Additional metadata
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, Value>,
}

impl ToolResult {
    /// Create a successful result
    pub fn success(data: Option<Value>) -> Self {
        Self {
            success: true,
            data,
            error: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a successful result with data
    pub fn success_with<T: serde::Serialize>(data: T) -> Self {
        Self {
            success: true,
            data: serde_json::to_value(data).ok(),
            error: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a failure result
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error.into()),
            metadata: HashMap::new(),
        }
    }

    /// Create a failure result with structured error data
    pub fn failure_with_data(error: impl Into<String>, data: Value) -> Self {
        Self {
            success: false,
            data: Some(data),
            error: Some(error.into()),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the result
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Trait for browser automation tools with associated parameter types
#[async_trait::async_trait]
pub trait Tool: Send + Sync + Default {
    /// Associated parameter type for this tool
    type Params: serde::Serialize + for<'de> serde::Deserialize<'de> + schemars::JsonSchema + Send;

    /// Get tool name
    fn name(&self) -> &str;

    /// Get tool parameter schema (JSON Schema)
    fn parameters_schema(&self) -> Value {
        serde_json::to_value(schemars::schema_for!(Self::Params)).unwrap_or_default()
    }

    /// Execute the tool with strongly-typed parameters
    async fn execute_typed(
        &self,
        params: Self::Params,
        context: &mut ToolContext<'_>,
    ) -> Result<ToolResult>;

    /// Execute the tool with JSON parameters (default implementation)
    async fn execute(&self, params: Value, context: &mut ToolContext<'_>) -> Result<ToolResult> {
        let typed_params: Self::Params = serde_json::from_value(params).map_err(|e| {
            crate::error::BrowserError::InvalidArgument(format!("Invalid parameters: {}", e))
        })?;
        self.execute_typed(typed_params, context).await
    }
}

/// Type-erased tool trait for dynamic dispatch
#[async_trait::async_trait]
pub trait DynTool: Send + Sync {
    fn name(&self) -> &str;
    fn parameters_schema(&self) -> Value;
    async fn execute(&self, params: Value, context: &mut ToolContext<'_>) -> Result<ToolResult>;
}

/// Blanket implementation to convert any Tool into DynTool
#[async_trait::async_trait]
impl<T: Tool> DynTool for T {
    fn name(&self) -> &str {
        Tool::name(self)
    }

    fn parameters_schema(&self) -> Value {
        Tool::parameters_schema(self)
    }

    async fn execute(&self, params: Value, context: &mut ToolContext<'_>) -> Result<ToolResult> {
        Tool::execute(self, params, context).await
    }
}

/// Tool registry for managing and accessing tools
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn DynTool>>,
}

impl ToolRegistry {
    /// Create a new empty tool registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Create a registry with default tools
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // Register the generate_image tool
        registry.register(image_generation::ImageGenerationTool);

        registry
    }

    /// Register a tool
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(tool));
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&Arc<dyn DynTool>> {
        self.tools.get(name)
    }

    /// Check if a tool exists
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// List all tool names
    pub fn list_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Get all tools
    pub fn all_tools(&self) -> Vec<Arc<dyn DynTool>> {
        self.tools.values().cloned().collect()
    }

    /// Execute a tool by name
    pub async fn execute(
        &self,
        name: &str,
        params: Value,
        context: &mut ToolContext<'_>,
    ) -> Result<ToolResult> {
        match self.get(name) {
            Some(tool) => tool.execute(params, context).await,
            None => Ok(ToolResult::failure(format!("Tool '{}' not found", name))),
        }
    }

    /// Get the number of registered tools
    pub fn count(&self) -> usize {
        self.tools.len()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success(Some(serde_json::json!({"url": "https://example.com"})));
        assert!(result.success);
        assert!(result.data.is_some());
        assert!(result.error.is_none());
    }

    #[test]
    fn test_tool_result_failure() {
        let result = ToolResult::failure("Test error");
        assert!(!result.success);
        assert!(result.data.is_none());
        assert_eq!(result.error, Some("Test error".to_string()));
    }

    #[test]
    fn test_tool_result_failure_with_data() {
        let result = ToolResult::failure_with_data(
            "Filtered",
            serde_json::json!({"type": "content_filter"}),
        );
        assert!(!result.success);
        assert_eq!(result.error, Some("Filtered".to_string()));
        assert_eq!(
            result.data,
            Some(serde_json::json!({"type": "content_filter"}))
        );
    }

    #[test]
    fn test_tool_result_with_metadata() {
        let result = ToolResult::success(None).with_metadata("duration_ms", serde_json::json!(100));

        assert!(result.metadata.contains_key("duration_ms"));
    }
}
