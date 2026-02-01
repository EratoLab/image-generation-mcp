# List available recipes by default.
default:
	just --list

# Run the MCP server.
run:
	cargo run --release --features mcp-server --bin mcp-server -- --port 51800

run_local:
	cargo run --release --features mcp-server --bin mcp-server -- --config mcp-server.local.toml --port 51800

check:
	cargo check --features mcp-server --bin mcp-server
