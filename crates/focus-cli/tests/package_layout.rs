//! Regression coverage for the shared CLI package layout.

use std::{fs, path::PathBuf};

#[test]
fn cli_implementation_is_shared_by_two_thin_binary_wrappers() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    assert!(manifest_dir.join("src/lib.rs").is_file());
    assert!(manifest_dir.join("src/main.rs").is_file());
    assert!(manifest_dir.join("src/bin/focus.rs").is_file());

    let manifest = fs::read_to_string(manifest_dir.join("Cargo.toml")).unwrap();
    assert!(manifest.contains("name = \"focus\""));
    assert!(!manifest.contains("name = \"focus\"\npath = \"src/main.rs\""));
}
