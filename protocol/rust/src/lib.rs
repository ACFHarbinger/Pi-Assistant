//! Pi-Protocol: IPC message types for Rust <-> Python communication.

use serde::{Deserialize, Serialize};

pub mod ipc;
pub mod ws;

pub use ipc::*;
pub use ws::*;
