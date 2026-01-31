//! Pi-Assistant: Universal Agent Harness
//!
//! Entry point for the Tauri v2 desktop application.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    pi_assistant_lib::run()
}
