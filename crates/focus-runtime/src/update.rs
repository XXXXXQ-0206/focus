//! Versioned Focus artifacts with hash verification and atomic activation.

use std::{
    env,
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

const MANIFEST_SCHEMA_VERSION: u32 = 1;
const MAX_LABEL_BYTES: usize = 128;
const HASH_HEX_BYTES: usize = 64;
const MAX_METADATA_BYTES: u64 = 64 * 1024;

/// A staged executable and its immutable verification metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateArtifact {
    /// Version directory label.
    pub version: String,
    /// Source revision used to produce the artifact.
    pub commit: String,
    /// Executable path relative to the update root.
    pub executable: PathBuf,
    /// Lowercase SHA-256 digest of the executable bytes.
    pub sha256: String,
    /// Executable byte length at staging time.
    pub size: u64,
}

/// Atomically published active and rollback artifact pointers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Artifact selected for the next supervisor launch.
    pub active: UpdateArtifact,
    /// Last active artifact retained for rollback.
    pub previous: Option<UpdateArtifact>,
}

/// Errors produced by the versioned artifact store.
#[derive(Debug, Error)]
pub enum UpdateError {
    /// Filesystem operation failed.
    #[error("update filesystem error: {0}")]
    Io(#[from] io::Error),
    /// Manifest serialization or validation failed.
    #[error("update manifest error: {0}")]
    Manifest(String),
    /// Artifact metadata no longer matches the staged bytes.
    #[error("update artifact verification failed: {0}")]
    Verification(String),
}

/// Filesystem owner for versioned artifacts and the active manifest.
#[derive(Debug, Clone)]
pub struct VersionedArtifactStore {
    root: PathBuf,
    manifest: PathBuf,
    staged: PathBuf,
}

impl VersionedArtifactStore {
    /// Create a store rooted at `root`; directories are created lazily.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            manifest: root.join("manifest.json"),
            staged: root.join("staged.json"),
            root,
        }
    }

    /// Return the manifest path for diagnostics and tests.
    #[must_use]
    pub fn manifest_path(&self) -> &Path {
        &self.manifest
    }

    /// Copy and hash one executable into its version directory.
    pub fn stage_file(
        &self,
        source: &Path,
        version: impl Into<String>,
        commit: impl Into<String>,
    ) -> Result<UpdateArtifact, UpdateError> {
        let version = validated_label(version.into(), "version")?;
        let commit = validated_label(commit.into(), "commit")?;
        let source_name = source
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| UpdateError::Manifest("source executable has no file name".into()))?;
        let target_dir = self.root.join("versions").join(&version);
        fs::create_dir_all(&target_dir)?;
        let target = target_dir.join(source_name);
        fs::copy(source, &target)?;
        let (sha256, size) = hash_file(&target)?;
        let executable = target
            .strip_prefix(&self.root)
            .map_err(|_| UpdateError::Manifest("staged executable escaped update root".into()))?
            .to_owned();
        let artifact = UpdateArtifact {
            version,
            commit,
            executable,
            sha256,
            size,
        };
        self.verify(&artifact)?;
        self.write_staged(&artifact)?;
        Ok(artifact)
    }

    /// Return the staged artifact selected by the last successful build.
    pub fn load_staged(&self) -> Result<Option<UpdateArtifact>, UpdateError> {
        if !self.staged.exists() {
            return Ok(None);
        }
        if fs::metadata(&self.staged)?.len() > MAX_METADATA_BYTES {
            return Err(UpdateError::Manifest(
                "staged metadata exceeds 64 KiB".into(),
            ));
        }
        let encoded = fs::read_to_string(&self.staged)?;
        let artifact: UpdateArtifact = serde_json::from_str(&encoded)
            .map_err(|error| UpdateError::Manifest(error.to_string()))?;
        self.verify(&artifact)?;
        Ok(Some(artifact))
    }

    /// Verify that an artifact is inside this store and matches its digest and size.
    pub fn verify(&self, artifact: &UpdateArtifact) -> Result<(), UpdateError> {
        validate_artifact(artifact)?;
        let path = self.root.join(&artifact.executable);
        let (sha256, size) = hash_file(&path)
            .map_err(|error| UpdateError::Verification(format!("{}: {error}", path.display())))?;
        if sha256 != artifact.sha256 || size != artifact.size {
            return Err(UpdateError::Verification(format!(
                "{} metadata does not match staged bytes",
                path.display()
            )));
        }
        Ok(())
    }

    /// Load and validate the active manifest, if one has been published.
    pub fn load_manifest(&self) -> Result<Option<UpdateManifest>, UpdateError> {
        if !self.manifest.exists() {
            return Ok(None);
        }
        if fs::metadata(&self.manifest)?.len() > MAX_METADATA_BYTES {
            return Err(UpdateError::Manifest(
                "update manifest exceeds 64 KiB".into(),
            ));
        }
        let encoded = fs::read_to_string(&self.manifest)?;
        let manifest: UpdateManifest = serde_json::from_str(&encoded)
            .map_err(|error| UpdateError::Manifest(error.to_string()))?;
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(UpdateError::Manifest(format!(
                "unsupported schema version {}",
                manifest.schema_version
            )));
        }
        self.verify(&manifest.active)?;
        if let Some(previous) = &manifest.previous {
            self.verify(previous)?;
        }
        Ok(Some(manifest))
    }

    /// Publish an artifact as active while retaining the prior active artifact.
    pub fn activate(&self, artifact: &UpdateArtifact) -> Result<UpdateManifest, UpdateError> {
        self.verify(artifact)?;
        let previous = self.load_manifest()?.map(|manifest| manifest.active);
        let manifest = UpdateManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            active: artifact.clone(),
            previous,
        };
        self.write_manifest(&manifest)?;
        let _ = fs::remove_file(&self.staged);
        Ok(manifest)
    }

    /// Select the previous artifact and clear the rollback slot.
    pub fn rollback(&self) -> Result<UpdateManifest, UpdateError> {
        let current = self
            .load_manifest()?
            .ok_or_else(|| UpdateError::Manifest("no active update manifest".into()))?;
        let active = current
            .previous
            .ok_or_else(|| UpdateError::Manifest("no previous artifact to roll back to".into()))?;
        self.verify(&active)?;
        let manifest = UpdateManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            active,
            previous: None,
        };
        self.write_manifest(&manifest)?;
        Ok(manifest)
    }

    fn write_staged(&self, artifact: &UpdateArtifact) -> Result<(), UpdateError> {
        fs::create_dir_all(&self.root)?;
        let temporary = self
            .staged
            .with_extension(format!("tmp-{}", Uuid::new_v4()));
        let encoded = serde_json::to_vec_pretty(artifact)
            .map_err(|error| UpdateError::Manifest(error.to_string()))?;
        fs::write(&temporary, encoded)?;
        if self.staged.exists() {
            let backup = self
                .staged
                .with_extension(format!("bak-{}", Uuid::new_v4()));
            fs::rename(&self.staged, &backup)?;
            if let Err(error) = fs::rename(&temporary, &self.staged) {
                let _ = fs::rename(&backup, &self.staged);
                let _ = fs::remove_file(&temporary);
                return Err(error.into());
            }
            let _ = fs::remove_file(backup);
        } else {
            fs::rename(&temporary, &self.staged)?;
        }
        Ok(())
    }

    fn write_manifest(&self, manifest: &UpdateManifest) -> Result<(), UpdateError> {
        fs::create_dir_all(&self.root)?;
        let temporary = self
            .manifest
            .with_extension(format!("tmp-{}", Uuid::new_v4()));
        let encoded = serde_json::to_vec_pretty(manifest)
            .map_err(|error| UpdateError::Manifest(error.to_string()))?;
        fs::write(&temporary, encoded)?;
        if self.manifest.exists() {
            let backup = self
                .manifest
                .with_extension(format!("bak-{}", Uuid::new_v4()));
            fs::rename(&self.manifest, &backup)?;
            if let Err(error) = fs::rename(&temporary, &self.manifest) {
                let _ = fs::rename(&backup, &self.manifest);
                let _ = fs::remove_file(&temporary);
                return Err(error.into());
            }
            let _ = fs::remove_file(backup);
        } else {
            fs::rename(&temporary, &self.manifest)?;
        }
        Ok(())
    }
}

/// Resolve the per-user update root used by the Focus supervisor.
#[must_use]
pub fn default_update_root() -> PathBuf {
    if let Some(path) = env::var_os("FOCUS_UPDATE_ROOT") {
        return PathBuf::from(path);
    }
    #[cfg(windows)]
    if let Some(root) = env::var_os("APPDATA") {
        return PathBuf::from(root).join("Focus").join("updates");
    }
    #[cfg(not(windows))]
    if let Some(root) = env::var_os("XDG_STATE_HOME") {
        return PathBuf::from(root).join("focus").join("updates");
    }
    env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".focus-updates")
}

fn validated_label(value: String, field: &str) -> Result<String, UpdateError> {
    if value.is_empty()
        || value.len() > MAX_LABEL_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(UpdateError::Manifest(format!(
            "{field} must contain only ASCII letters, digits, '.', '_' or '-'; max {MAX_LABEL_BYTES} bytes"
        )));
    }
    Ok(value)
}

fn validate_artifact(artifact: &UpdateArtifact) -> Result<(), UpdateError> {
    validated_label(artifact.version.clone(), "version")?;
    validated_label(artifact.commit.clone(), "commit")?;
    if artifact.sha256.len() != HASH_HEX_BYTES
        || !artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(UpdateError::Verification("invalid SHA-256 digest".into()));
    }
    if artifact.executable.is_absolute()
        || artifact
            .executable
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(UpdateError::Verification(
            "artifact executable must stay inside update root".into(),
        ));
    }
    let expected_prefix = Path::new("versions").join(&artifact.version);
    if !artifact.executable.starts_with(&expected_prefix) {
        return Err(UpdateError::Verification(
            "artifact executable must live in its version directory".into(),
        ));
    }
    Ok(())
}

/// Hash a file with SHA-256 and return `(lowercase_hex, byte_length)`.
pub fn hash_file(path: &Path) -> Result<(String, u64), io::Error> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut size = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        size = size.saturating_add(read as u64);
    }
    Ok((format!("{:x}", hasher.finalize()), size))
}

#[cfg(test)]
mod tests {
    use super::{UpdateArtifact, UpdateManifest, VersionedArtifactStore};
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn fixture() -> (PathBuf, VersionedArtifactStore, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "focus-update-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let store = VersionedArtifactStore::new(root.join("updates"));
        let source = root.join("focus.exe");
        fs::write(&source, b"focus-v1").unwrap();
        (root, store, source)
    }

    #[test]
    fn stages_hashes_and_verifies_a_versioned_artifact() {
        let (root, store, source) = fixture();
        let artifact = store.stage_file(&source, "v1", "abc123").unwrap();
        assert_eq!(artifact.version, "v1");
        assert_eq!(artifact.commit, "abc123");
        assert_eq!(artifact.size, 8);
        assert_eq!(artifact.sha256.len(), 64);
        store.verify(&artifact).unwrap();
        assert_eq!(store.load_staged().unwrap(), Some(artifact));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn activation_and_rollback_restore_the_previous_manifest() {
        let (root, store, source) = fixture();
        let first = store.stage_file(&source, "v1", "one").unwrap();
        store.activate(&first).unwrap();
        fs::write(&source, b"focus-v2").unwrap();
        let second = store.stage_file(&source, "v2", "two").unwrap();
        let active = store.activate(&second).unwrap();
        assert_eq!(active.active.version, "v2");
        assert_eq!(active.previous.as_ref().unwrap().version, "v1");
        assert!(store.load_staged().unwrap().is_none());
        let rolled_back = store.rollback().unwrap();
        assert_eq!(rolled_back.active.version, "v1");
        assert!(rolled_back.previous.is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn manifest_round_trip_rejects_artifacts_outside_the_store() {
        let (root, store, source) = fixture();
        let artifact = store.stage_file(&source, "v1", "one").unwrap();
        store.activate(&artifact).unwrap();
        let mut raw = fs::read_to_string(store.manifest_path()).unwrap();
        raw = raw.replace("focus.exe", "../outside.exe");
        fs::write(store.manifest_path(), raw).unwrap();
        let error = store.load_manifest().unwrap_err().to_string();
        assert!(error.contains("inside update root"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn manifest_types_keep_schema_and_artifact_fields_stable() {
        let artifact = UpdateArtifact {
            version: "v1".into(),
            commit: "abc".into(),
            executable: PathBuf::from("versions/v1/focus.exe"),
            sha256: "0".repeat(64),
            size: 1,
        };
        let manifest = UpdateManifest {
            schema_version: 1,
            active: artifact.clone(),
            previous: Some(artifact),
        };
        let encoded = serde_json::to_string(&manifest).unwrap();
        let decoded: UpdateManifest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.schema_version, 1);
        assert_eq!(decoded.active.version, "v1");
    }
}
