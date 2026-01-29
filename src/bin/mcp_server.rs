//! Image Generation MCP Server
//!
//! This binary provides a Model Context Protocol (MCP) server for image generation.
//! It uses configurable backends to generate images from text prompts.

use image_generation_mcp::mcp::BrowserServer;
use clap::{Parser, ValueEnum};
use log::{debug, info};
use rmcp::{ServiceExt, transport::stdio};
use std::io::{stdin, stdout};
use std::sync::Arc;

#[cfg(feature = "mcp-server")]
use rmcp::transport::{
    sse_server::{SseServer, SseServerConfig},
    streamable_http_server::{StreamableHttpService, session::local::LocalSessionManager},
};

#[cfg(feature = "mcp-server")]
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Transport {
    /// Standard input/output transport (default)
    Stdio,
    /// Server-Sent Events transport
    Sse,
    /// HTTP streamable transport
    Http,
}

#[derive(Parser)]
#[command(name = "image-generation-mcp")]
#[command(version)]
#[command(about = "Image Generation MCP server with configurable backends", long_about = None)]
struct Cli {
    /// Path to MCP server configuration file
    #[arg(long, short = 'c')]
    config: Option<String>,

    /// Transport type to use (overrides config file)
    #[arg(long, short = 't', value_enum)]
    transport: Option<Transport>,

    /// Port for SSE or HTTP transport (overrides config file)
    #[arg(long, short = 'p')]
    port: Option<u16>,

    /// Host address to bind to (overrides config file)
    #[arg(long)]
    host: Option<String>,

    /// SSE endpoint path (overrides config file)
    #[arg(long)]
    sse_path: Option<String>,

    /// SSE POST path for messages (overrides config file)
    #[arg(long)]
    sse_post_path: Option<String>,

    /// HTTP streamable endpoint path (overrides config file)
    #[arg(long)]
    http_path: Option<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("Image Generation MCP Server v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = Arc::new(
        image_generation_mcp::config::ServerConfig::load(cli.config).await?
    );

    // Determine final transport settings (CLI overrides config file)
    let transport_type = cli.transport
        .map(|t| match t {
            Transport::Stdio => "stdio".to_string(),
            Transport::Sse => "sse".to_string(),
            Transport::Http => "http".to_string(),
        })
        .unwrap_or_else(|| config.transport.transport.clone());

    let host = cli.host.unwrap_or_else(|| config.transport.host.clone());
    let port = cli.port.unwrap_or(config.transport.port);
    let sse_path = cli.sse_path.unwrap_or_else(|| config.transport.sse_path.clone());
    let sse_post_path = cli.sse_post_path.unwrap_or_else(|| config.transport.sse_post_path.clone());
    let http_path = cli.http_path.unwrap_or_else(|| config.transport.http_path.clone());

    info!("Configuration loaded: tool type = {}", match &config.tool {
        image_generation_mcp::config::ToolSection::Mock(_) => "mock",
    });
    info!("Transport: {} ({})", transport_type,
        match transport_type.as_str() {
            "stdio" => "standard input/output".to_string(),
            "sse" => format!("{}:{}", host, port),
            "http" => format!("{}:{}", host, port),
            _ => "unknown".to_string(),
        }
    );

    // Set global config for access by tool handlers
    image_generation_mcp::config::set_global_config(config).await;

    // Route to appropriate transport
    match transport_type.as_str() {
        "stdio" => {
            info!("Ready to accept MCP connections via stdio");
            let (_read, _write) = (stdin(), stdout());
            let service = BrowserServer::new()
                .map_err(|e| format!("Failed to create image generation server: {}", e))?;
            let server = service.serve(stdio()).await?;

            // Set up signal handler for graceful shutdown
            #[cfg(unix)]
            {
                let mut sigterm =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
                let mut sigint =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;

                tokio::select! {
                    quit_reason = server.waiting() => {
                        debug!("Server quit with reason: {:?}", quit_reason);
                    }
                    _ = sigterm.recv() => {
                        info!("Received SIGTERM, shutting down gracefully...");
                    }
                    _ = sigint.recv() => {
                        info!("Received SIGINT (Ctrl+C), shutting down gracefully...");
                    }
                }
            }

            #[cfg(windows)]
            {
                let mut ctrl_c = tokio::signal::windows::ctrl_c()?;
                let mut ctrl_break = tokio::signal::windows::ctrl_break()?;

                tokio::select! {
                    quit_reason = server.waiting() => {
                        debug!("Server quit with reason: {:?}", quit_reason);
                    }
                    _ = ctrl_c.recv() => {
                        info!("Received Ctrl+C, shutting down gracefully...");
                    }
                    _ = ctrl_break.recv() => {
                        info!("Received Ctrl+Break, shutting down gracefully...");
                    }
                }
            }

            #[cfg(not(any(unix, windows)))]
            {
                let quit_reason = server.waiting().await;
                debug!("Server quit with reason: {:?}", quit_reason);
            }
        }
        "sse" => {
            info!("Host: {}", host);
            info!("Port: {}", port);
            info!("SSE path: {}", sse_path);
            info!("SSE POST path: {}", sse_post_path);

            let bind_addr = format!("{}:{}", host, port);

            // Create SSE server configuration
            let sse_config = SseServerConfig {
                bind: bind_addr.parse()?,
                sse_path: sse_path.clone(),
                post_path: sse_post_path.clone(),
                ct: CancellationToken::new(),
                sse_keep_alive: None,
            };

            // Create SSE server and router
            let (sse_server, router) = SseServer::new(sse_config);

            info!(
                "Ready to accept MCP connections at http://{}{}",
                bind_addr, sse_path
            );

            // Register service factory for each connection
            let _cancellation_token = sse_server.with_service(move || {
                BrowserServer::new()
                    .expect("Failed to create image generation server")
            });

            // Start HTTP server with SSE router
            let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
            axum::serve(listener, router.into_make_service()).await?;
        }
        "http" => {
            info!("HTTP streamable");
            info!("Host: {}", host);
            info!("Port: {}", port);
            info!("HTTP path: {}", http_path);

            let bind_addr = format!("{}:{}", host, port);

            // Create service factory closure
            let service_factory = move || {
                BrowserServer::new()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
            };

            let http_service = StreamableHttpService::new(
                service_factory,
                LocalSessionManager::default().into(),
                Default::default(),
            );

            let router = axum::Router::new().nest_service(&http_path, http_service);

            info!(
                "Ready to accept MCP connections at http://{}{}",
                bind_addr, http_path
            );

            let listener = tokio::net::TcpListener::bind(bind_addr).await?;
            axum::serve(listener, router).await?;
        }
        _ => {
            return Err(format!("Invalid transport type: {}", transport_type).into());
        }
    }

    Ok(())
}
