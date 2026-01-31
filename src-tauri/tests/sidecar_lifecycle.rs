use pi_assistant_lib::ipc::sidecar::SidecarHandle;
use serde_json::json;

#[tokio::test]
async fn test_sidecar_spawn_and_ping() {
    // Adjusting for project structure:
    // Sidecar is in ../sidecar relative to src-tauri root
    // We should probably rely on `python3` being available OR the venv python.
    // Let's assume the sidecar venv is at `../sidecar/.venv/bin/python`

    let python_path = std::env::current_dir()
        .unwrap()
        .parent()
        .unwrap()
        .join("sidecar/.venv/bin/python");

    // Ensure python exists (if not, maybe skip or fail)
    if !python_path.exists() {
        eprintln!("Skipping test: Python venv not found at {:?}", python_path);
        return;
    }

    let mut sidecar = SidecarHandle::new()
        .with_python_path(python_path.to_str().unwrap())
        .with_sidecar_module("pi_sidecar");

    // Start
    sidecar.start().await.expect("Failed to start sidecar");
    assert!(sidecar.is_alive());

    // Ping
    let result = sidecar
        .request("health.ping", json!({}))
        .await
        .expect("Ping failed");
    assert_eq!(result["status"], "ok");

    // Stop
    sidecar.stop().await.expect("Failed to stop sidecar");

    // Verify stopped
    assert!(!sidecar.is_alive());
}
