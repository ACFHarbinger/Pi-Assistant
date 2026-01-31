//! Safety and permission module.

pub mod permission;
pub mod rules;

pub use permission::{PermissionEngine, PermissionResult};
pub use rules::PermissionRules;
