fn main() {
    let project_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-search=native={}/libs", project_dir);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/libs", project_dir);
    tauri_build::build()
}
