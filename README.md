# image-generation-mcp

A Model Context Protocol (MCP) server for image generation with configurable backends.

## ✨ Highlights

- **Configurable Backends** - Switch between image generation providers via TOML configuration
- **Token Usage Tracking** - Built-in tracking with `_meta` fields for monitoring usage
- **Extensible Architecture** - Easy to add new backends (mock, OpenAI, Stability AI, etc.)
- **MCP Compatible** - Standard MCP server implementation for AI assistants
- **Simple Configuration** - TOML-based configuration with flexible path resolution

## Installation

```bash
cargo add image-generation-mcp
```

## Quick Start

### Running the MCP Server

Create a configuration file `mcp-server.toml`:

```toml
[server]
name = "image-generation-mcp"
version = "0.1.0"

[tool]
type = "mock"

[tool.mock]
static_image_path = "static/image_1.png"

[metadata]
prefix = "chat.erato/"
```

Run the server:

```bash
# Using default config path (./mcp-server.toml)
cargo run --bin mcp-server --features mcp-server

# Using custom config path
cargo run --bin mcp-server --features mcp-server -- --config /path/to/config.toml

# Using environment variable
export MCP_SERVER_CONFIG=/path/to/config.toml
cargo run --bin mcp-server --features mcp-server

# SSE transport
cargo run --bin mcp-server --features mcp-server -- --transport sse --host 0.0.0.0 --port 3000

# HTTP transport
cargo run --bin mcp-server --features mcp-server -- --transport http --host 0.0.0.0 --port 3000
```

## Configuration

The server supports configuration via TOML file with flexible path resolution:

**Priority Order:**
1. CLI argument: `--config <path>` (highest priority)
2. Environment variable: `MCP_SERVER_CONFIG`
3. Default path: `./mcp-server.toml` (lowest priority)

### Configuration File Structure

```toml
[server]
name = "image-generation-mcp"
version = "0.1.0"

[tool]
type = "mock"  # Backend type: mock, openai, stability-ai, etc.

# Mock backend configuration
[tool.mock]
static_image_path = "static/image_1.png"

[metadata]
prefix = "chat.erato/"  # Prefix for token usage metadata fields

# Transport configuration
# All settings can be overridden via CLI arguments
[transport]
# Transport type: "stdio", "sse", or "http"
transport = "stdio"

# For SSE and HTTP transports:
# host = "127.0.0.1"
# port = 3000
# sse_path = "/sse"
# sse_post_path = "/message"
# http_path = "/mcp"
```

### Transport Configuration

The server supports three transport types that can be configured in the TOML file or overridden via CLI arguments:

**Transport Types:**
- **stdio** (default): Standard input/output for local communication
- **sse**: Server-Sent Events for web-based connections
- **http**: HTTP streamable transport

**Configuration Priority:**
CLI arguments > Config file settings > Defaults

Example with CLI override:
```bash
# Config file has transport = "stdio", but CLI overrides to SSE
cargo run --release --features mcp-server --bin mcp-server -- \
  --transport sse \
  --host 0.0.0.0 \
  --port 3000
```

### Backend Configuration

#### Mock Backend (Default)
Returns a static image file. Useful for testing and development.

```toml
[tool]
type = "mock"

[tool.mock]
static_image_path = "path/to/image.png"
```

## MCP Tool

The server exposes a single MCP tool:

### `generate_image`

Generate images from text prompts.

**Parameters:**
- `prompt` (string, required): Text prompt for image generation
- `width` (integer, optional): Width of generated image [default: 1024]
- `height` (integer, optional): Height of generated image [default: 1024]
- `num_images` (integer, optional): Number of images to generate [default: 1]

**Returns:**
- `images` (array): Array of generated image objects with:
  - `data_uri` (string): Base64-encoded data URI (e.g., `data:image/png;base64,...`)
  - `width` (integer): Image width in pixels
  - `height` (integer): Image height in pixels
  - `mime_type` (string): MIME type (e.g., `image/png`)

**Token Usage Metadata:**

The response includes `_meta` fields with token usage tracking:
- `chat.erato/token_usage/total_prompt_tokens`: Tokens used for the prompt
- `chat.erato/token_usage/total_completion_tokens`: Tokens used for generation
- `chat.erato/token_usage/total_tokens`: Total tokens used
- `chat.erato/token_usage/model_identifier`: Model/backend identifier

**Example Request:**

```json
{
  "prompt": "A serene mountain landscape at sunset",
  "width": 1024,
  "height": 1024,
  "num_images": 1
}
```

**Example Response:**

```json
{
  "images": [
    {
      "data_uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
      "width": 500,
      "height": 500,
      "mime_type": "image/png"
    }
  ],
  "_meta": {
    "chat.erato/token_usage/total_prompt_tokens": 120,
    "chat.erato/token_usage/total_completion_tokens": 0,
    "chat.erato/token_usage/total_tokens": 120,
    "chat.erato/token_usage/model_identifier": "mock/v1"
  }
}
```

## Extensibility

### Adding New Backends

The architecture makes it easy to add new image generation backends:

1. **Add a new variant to the `ToolSection` enum** in `src/config.rs`:

```rust
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolSection {
    Mock(MockToolConfig),
    OpenAI(OpenAIToolConfig),  // New
    StabilityAI(StabilityAIToolConfig),  // New
}
```

2. **Create the config struct** in `src/config.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolConfig {
    pub api_key: String,
    pub model: String,
}
```

3. **Add generation logic** in `src/tools/image_generation.rs`:

```rust
match &config.tool {
    ToolSection::Mock(mock_config) => {
        // Mock logic
    }
    ToolSection::OpenAI(openai_config) => {
        // Call OpenAI API
    }
}
```

4. **Update configuration documentation** with examples

## CLI Options

All CLI options override values from the configuration file:

```bash
Options:
  -c, --config <PATH>         Path to MCP server configuration file
  -t, --transport <TYPE>      Transport type (overrides config): stdio, sse, http
  -p, --port <PORT>          Port for SSE or HTTP transport (overrides config)
      --host <HOST>          Host address to bind to (overrides config)
      --sse-path <PATH>      SSE endpoint path (overrides config)
      --sse-post-path <PATH> SSE POST path for messages (overrides config)
      --http-path <PATH>     HTTP streamable endpoint path (overrides config)
  -h, --help                 Print help
  -V, --version              Print version
```

**Example:**
```bash
# Use transport from config file (stdio)
cargo run --release --features mcp-server --bin mcp-server

# Override config file transport with SSE
cargo run --release --features mcp-server --bin mcp-server -- \
  --transport sse --host 0.0.0.0 --port 4000
```

## Docker

Build and run with Docker:

```bash
# Build the image
docker build -t image-generation-mcp .

# Run the container
docker run -p 3000:3000 \
  -v $(pwd)/mcp-server.toml:/app/mcp-server.toml \
  -v $(pwd)/static:/app/static \
  image-generation-mcp

# With custom config
docker run -p 3000:3000 \
  -v /path/to/config.toml:/app/mcp-server.toml \
  image-generation-mcp
```

## Requirements

- Rust 1.70+
- Configuration file (see examples above)

## Token Usage Tracking

The server tracks token usage for each tool call and returns it via `_meta` fields:

- **Mock Backend**: Approximates tokens using `prompt_length × 4`
- **Real Backends**: Use actual token counts from the API

All token usage is tracked **per-tool-call** (not per-server instance), ensuring accurate usage tracking for each request.

## Architecture

```
src/
├── config.rs           # Configuration structures and loading
├── error.rs            # Error types
├── lib.rs              # Library root
├── bin/
│   └── mcp_server.rs   # Server binary with transport handling
├── mcp/
│   ├── mod.rs          # MCP tool registration macro
│   └── handler.rs      # ServerHandler implementation
└── tools/
    ├── mod.rs          # Tool traits and registry
    └── image_generation.rs  # Image generation tool
```

## License

MIT
