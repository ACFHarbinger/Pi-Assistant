//! Pi-Assistant: Universal Agent Harness
//!
//! Entry point for the Tauri v2 desktop application.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    if let Some(home) = dirs::home_dir() {
        let dot_dir = home.join(".pi-assistant");
        std::env::set_var("TAURI_APP_DATA_DIR", &dot_dir);
        std::env::set_var("XDG_DATA_HOME", &dot_dir);
    }
    pi_assistant_lib::run()
}
