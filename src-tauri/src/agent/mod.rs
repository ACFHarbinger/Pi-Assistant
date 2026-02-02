//! Agent loop and execution system.

pub mod cost;
pub mod executor;
pub mod r#loop;
pub mod monitor;
pub mod planner;
pub mod pool;
pub mod transaction;

pub use executor::AgentExecutor;
pub use monitor::spawn_agent_coordinator;
pub use planner::AgentPlanner;
pub use pool::{AgentInstance, AgentPool, RoutingConfig};
pub use r#loop::{spawn_agent_loop, AgentLoopHandle, AgentPlan, AgentTask};
pub use transaction::{TransactionManager, UndoableAction};
