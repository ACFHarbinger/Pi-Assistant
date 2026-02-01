//! Pi-Core: Shared types and logic (no Tauri dependency).

pub mod agent_types;
pub mod protocol;
pub mod task_manager;

use wasm_bindgen::prelude::*;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}
