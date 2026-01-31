//! WebSocket server for mobile client communication.

pub mod handler;
pub mod server;

pub use handler::WsHandler;
pub use server::WebSocketServer;
