//! Executable release dependency metadata gate.

use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use serde_json::json;

const RIDER_HEADING: &str = "MIT License (with OpenAI/Anthropic Rider)";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateMode {
    Technical,
    Production,
}

impl GateMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "technical" => Ok(Self::Technical),
            "production" => Ok(Self::Production),
            _ => Err(format!("invalid release mode `{value}`")),
        }
    }

    fn allows(self, audit: &Audit) -> bool {
        self == Self::Technical || !audit.review_required
    }
}

#[derive(Debug)]
struct Config {
    mode: GateMode,
    workspace: PathBuf,
    registry_root: Option<PathBuf>,
    inventory: Option<PathBuf>,
}

impl Config {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut arguments = arguments.into_iter();
        let mut mode = GateMode::Production;
        let mut workspace = env::current_dir().map_err(|error| error.to_string())?;
        let mut registry_root = None;
        let mut inventory = None;
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--mode" => {
                    mode = GateMode::parse(
                        &arguments
                            .next()
                            .ok_or_else(|| "--mode requires a value".to_owned())?,
                    )?;
                }
                "--workspace" => {
                    workspace = PathBuf::from(
                        arguments
                            .next()
                            .ok_or_else(|| "--workspace requires a path".to_owned())?,
                    );
                }
                "--registry-root" => {
                    registry_root =
                        Some(PathBuf::from(arguments.next().ok_or_else(|| {
                            "--registry-root requires a path".to_owned()
                        })?));
                }
                "--inventory" => {
                    inventory = Some(PathBuf::from(
                        arguments
                            .next()
                            .ok_or_else(|| "--inventory requires a path".to_owned())?,
                    ));
                }
                "--help" | "-h" => return Err(usage()),
                value => return Err(format!("unexpected argument `{value}`")),
            }
        }
        Ok(Self {
            mode,
            workspace,
            registry_root,
            inventory,
        })
    }
}

#[derive(Debug, Clone)]
struct LockPackage {
    name: String,
    version: String,
    source: Option<String>,
    checksum: Option<String>,
}

#[derive(Debug, Clone)]
enum LicenseMetadata {
    Spdx(String),
    File { path: String, heading: String },
    Missing(String),
}

#[derive(Debug, Clone)]
struct PackageRecord {
    name: String,
    version: String,
    direct: bool,
    source: Option<String>,
    checksum: Option<String>,
    metadata: LicenseMetadata,
    rider: bool,
    evidence: String,
}

impl PackageRecord {
    fn from_lock(
        package: LockPackage,
        direct: bool,
        metadata: LicenseMetadata,
        rider: bool,
        evidence: String,
    ) -> Self {
        Self {
            name: package.name,
            version: package.version,
            direct,
            source: package.source,
            checksum: package.checksum,
            metadata,
            rider,
            evidence,
        }
    }

    #[cfg(test)]
    fn licensed(name: &str, version: &str, license: &str) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            direct: false,
            source: None,
            checksum: None,
            metadata: LicenseMetadata::Spdx(license.into()),
            rider: false,
            evidence: "test package metadata".into(),
        }
    }

    #[cfg(test)]
    fn license_file(name: &str, version: &str, heading: &str) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            direct: false,
            source: None,
            checksum: None,
            metadata: LicenseMetadata::File {
                path: "LICENSE".into(),
                heading: heading.into(),
            },
            rider: heading.contains("OpenAI/Anthropic Rider"),
            evidence: "test license file".into(),
        }
    }

    #[cfg(test)]
    fn missing(name: &str, version: &str) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            direct: false,
            source: None,
            checksum: None,
            metadata: LicenseMetadata::Missing("missing metadata".into()),
            rider: false,
            evidence: "test package metadata".into(),
        }
    }
}

#[derive(Debug)]
struct Audit {
    packages: Vec<PackageRecord>,
    review_required: bool,
    metadata_issues: usize,
    rider_packages: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProductionBlocker {
    package: String,
    version: String,
    reason: String,
    evidence: String,
}

fn audit_packages(packages: &[PackageRecord]) -> Audit {
    let metadata_issues = packages
        .iter()
        .filter(|package| metadata_has_issue(&package.metadata))
        .count();
    let rider_packages = packages
        .iter()
        .filter(|package| package.rider || metadata_has_rider(&package.metadata))
        .count();
    Audit {
        packages: packages.to_vec(),
        review_required: metadata_issues > 0 || rider_packages > 0,
        metadata_issues,
        rider_packages,
    }
}

fn package_blocking_reasons(package: &PackageRecord) -> Vec<String> {
    let mut reasons = Vec::new();
    if let LicenseMetadata::Missing(reason) = &package.metadata {
        reasons.push(format!("license metadata unresolved: {reason}"));
    }
    if package.rider || metadata_has_rider(&package.metadata) {
        reasons.push(
            "license contains an OpenAI/Anthropic Rider or LicenseRef restriction requiring an explicit production release decision".into(),
        );
    }
    reasons
}

fn package_evidence(package: &PackageRecord) -> String {
    let source = package.source.as_deref().unwrap_or("<none>");
    let checksum = package.checksum.as_deref().unwrap_or("<none>");
    format!(
        "{}; Cargo.lock source={source}; checksum={checksum}",
        package.evidence
    )
}

fn production_blockers(audit: &Audit) -> Vec<ProductionBlocker> {
    audit
        .packages
        .iter()
        .flat_map(|package| {
            let evidence = package_evidence(package);
            package_blocking_reasons(package)
                .into_iter()
                .map(move |reason| ProductionBlocker {
                    package: package.name.clone(),
                    version: package.version.clone(),
                    reason,
                    evidence: evidence.clone(),
                })
        })
        .collect()
}

fn validate_kernel_dependency_boundary(audit: &Audit) -> Result<(), String> {
    if audit
        .packages
        .iter()
        .any(|package| package.name == "pi_agent_rust")
    {
        return Err("legacy pi_agent_rust dependency remains in Cargo.lock".into());
    }
    Ok(())
}

fn metadata_has_rider(metadata: &LicenseMetadata) -> bool {
    match metadata {
        LicenseMetadata::Spdx(value) => {
            value.contains("OpenAI") || value.contains("Anthropic") || value.contains("Rider")
        }
        LicenseMetadata::File { heading, .. } => {
            heading.contains("OpenAI") || heading.contains("Anthropic") || heading.contains("Rider")
        }
        LicenseMetadata::Missing(_) => false,
    }
}

fn metadata_has_issue(metadata: &LicenseMetadata) -> bool {
    match metadata {
        LicenseMetadata::Spdx(value) => value.trim().is_empty() || value.contains('/'),
        LicenseMetadata::File { .. } => false,
        LicenseMetadata::Missing(_) => true,
    }
}

fn normalize_legacy_slash_spdx(value: &str) -> Option<String> {
    const KNOWN_IDS: &[&str] = &["Apache-2.0", "MIT", "Unlicense"];

    let terms = value.split('/').map(str::trim).collect::<Vec<_>>();
    (terms.len() > 1
        && terms
            .iter()
            .all(|term| !term.is_empty() && KNOWN_IDS.contains(term)))
    .then(|| terms.join(" OR "))
}

fn parse_quoted(line: &str, key: &str) -> Option<String> {
    let value = line
        .strip_prefix(key)?
        .trim_start()
        .strip_prefix('=')?
        .trim();
    let value = value.strip_prefix('"')?;
    let mut escaped = false;
    let mut output = String::new();
    for character in value.chars() {
        if escaped {
            output.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == '"' {
            return Some(output);
        } else {
            output.push(character);
        }
    }
    None
}

fn parse_lock(content: &str) -> Result<Vec<LockPackage>, String> {
    let mut packages = Vec::new();
    let mut current: Option<LockPackage> = None;
    for line in content.lines().map(str::trim) {
        if line == "[[package]]" {
            if let Some(package) = current.take() {
                packages.push(package);
            }
            current = Some(LockPackage {
                name: String::new(),
                version: String::new(),
                source: None,
                checksum: None,
            });
            continue;
        }
        let Some(package) = current.as_mut() else {
            continue;
        };
        if let Some(value) = parse_quoted(line, "name") {
            package.name = value;
        } else if let Some(value) = parse_quoted(line, "version") {
            package.version = value;
        } else if let Some(value) = parse_quoted(line, "source") {
            package.source = Some(value);
        } else if let Some(value) = parse_quoted(line, "checksum") {
            package.checksum = Some(value);
        }
    }
    if let Some(package) = current {
        packages.push(package);
    }
    if packages
        .iter()
        .any(|package| package.name.is_empty() || package.version.is_empty())
    {
        return Err("Cargo.lock contains an incomplete package record".into());
    }
    Ok(packages)
}

#[derive(Debug, Default)]
struct ManifestLicense {
    license: Option<String>,
    license_file: Option<String>,
}

fn parse_manifest_license(content: &str) -> ManifestLicense {
    let mut package_section = false;
    let mut metadata = ManifestLicense::default();
    for line in content.lines().map(str::trim) {
        if line.starts_with('[') {
            package_section = line == "[package]";
            continue;
        }
        if !package_section {
            continue;
        }
        if let Some(value) = parse_quoted(line, "license") {
            metadata.license = Some(value);
        } else if let Some(value) = parse_quoted(line, "license-file") {
            metadata.license_file = Some(value);
        }
    }
    metadata
}

fn registry_roots(override_root: Option<&Path>) -> Result<Vec<PathBuf>, String> {
    if let Some(root) = override_root {
        return Ok(vec![root.to_path_buf()]);
    }
    let cargo_home = env::var_os("CARGO_HOME").map(PathBuf::from).or_else(|| {
        env::var_os("USERPROFILE")
            .or_else(|| env::var_os("HOME"))
            .map(|home| PathBuf::from(home).join(".cargo"))
    });
    let source_root = cargo_home
        .ok_or_else(|| "CARGO_HOME and user home are unavailable".to_owned())?
        .join("registry")
        .join("src");
    let mut roots = fs::read_dir(&source_root)
        .map_err(|error| format!("failed to read {}: {error}", source_root.display()))?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    roots.sort();
    if roots.is_empty() {
        return Err(format!(
            "no registry sources below {}",
            source_root.display()
        ));
    }
    Ok(roots)
}

fn collect_direct_names(workspace: &Path) -> Result<BTreeSet<String>, String> {
    let mut manifests = vec![workspace.join("Cargo.toml")];
    let crates = workspace.join("crates");
    if crates.is_dir() {
        for entry in fs::read_dir(&crates)
            .map_err(|error| format!("failed to read {}: {error}", crates.display()))?
        {
            let path = entry
                .map_err(|error| error.to_string())?
                .path()
                .join("Cargo.toml");
            if path.is_file() {
                manifests.push(path);
            }
        }
    }
    let mut names = BTreeSet::new();
    for manifest in manifests {
        let content = fs::read_to_string(&manifest)
            .map_err(|error| format!("failed to read {}: {error}", manifest.display()))?;
        let mut dependency_section = false;
        for line in content.lines().map(str::trim) {
            if line.starts_with('[') {
                dependency_section = line.contains("dependencies");
                continue;
            }
            if !dependency_section || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            if !key.is_empty()
                && key
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || "_-".contains(character))
            {
                names.insert(key.to_owned());
                if let Some(package_position) = value.find("package")
                    && let Some(package) = parse_quoted(&value[package_position..], "package")
                {
                    names.insert(package);
                }
            }
        }
    }
    Ok(names)
}

fn locate_registry_package(roots: &[PathBuf], package: &LockPackage) -> Option<PathBuf> {
    let directory = format!("{}-{}", package.name, package.version);
    roots
        .iter()
        .map(|root| root.join(&directory))
        .find(|path| path.is_dir())
}

fn inspect_packages(
    lock_packages: Vec<LockPackage>,
    direct_names: &BTreeSet<String>,
    roots: &[PathBuf],
) -> Vec<PackageRecord> {
    let mut records = Vec::new();
    for package in lock_packages
        .into_iter()
        .filter(|package| package.source.is_some())
    {
        let direct = direct_names.contains(&package.name);
        let cache_entry = format!("{}-{}", package.name, package.version);
        let Some(directory) = locate_registry_package(roots, &package) else {
            records.push(PackageRecord::from_lock(
                package,
                direct,
                LicenseMetadata::Missing("registry source is not cached".into()),
                false,
                format!("Cargo registry source cache is missing `{cache_entry}`"),
            ));
            continue;
        };
        let manifest_path = directory.join("Cargo.toml");
        let manifest = fs::read_to_string(&manifest_path);
        let metadata = match manifest {
            Ok(content) => parse_manifest_license(&content),
            Err(error) => {
                records.push(PackageRecord::from_lock(
                    package,
                    direct,
                    LicenseMetadata::Missing(format!("failed to read registry manifest: {error}")),
                    false,
                    format!("registry manifest `{}`", manifest_path.display()),
                ));
                continue;
            }
        };
        let manifest_evidence =
            format!("registry manifest `{}` [package]", manifest_path.display());
        let (license_metadata, rider, evidence) = if let Some(license) = metadata.license {
            if license.trim().is_empty() {
                (
                    LicenseMetadata::Missing("package has an empty license field".into()),
                    false,
                    format!("{manifest_evidence}.license"),
                )
            } else if let Some(normalized) = normalize_legacy_slash_spdx(&license) {
                (
                    LicenseMetadata::Spdx(normalized),
                    false,
                    format!(
                        "{manifest_evidence}.license; normalized known legacy slash expression `{license}`"
                    ),
                )
            } else if license.contains('/') {
                (
                    LicenseMetadata::Missing(format!(
                        "license field is not an SPDX expression: {license}"
                    )),
                    false,
                    format!("{manifest_evidence}.license"),
                )
            } else {
                let rider = license.contains("OpenAI")
                    || license.contains("Anthropic")
                    || license.contains("Rider");
                (
                    LicenseMetadata::Spdx(license),
                    rider,
                    format!("{manifest_evidence}.license"),
                )
            }
        } else if let Some(path) = metadata.license_file {
            let license_path = directory.join(&path);
            match fs::read_to_string(&license_path) {
                Ok(content) => {
                    let heading = content
                        .lines()
                        .map(str::trim)
                        .find(|line| !line.is_empty())
                        .unwrap_or("empty license file")
                        .to_owned();
                    let rider = content.contains(RIDER_HEADING)
                        || content.contains("ADDITIONAL RIDER / RESTRICTION");
                    (
                        LicenseMetadata::File { path, heading },
                        rider,
                        format!(
                            "{manifest_evidence}.license-file -> `{}`",
                            license_path.display()
                        ),
                    )
                }
                Err(error) => (
                    LicenseMetadata::Missing(format!("failed to read license file: {error}")),
                    false,
                    format!(
                        "{manifest_evidence}.license-file -> `{}`",
                        license_path.display()
                    ),
                ),
            }
        } else {
            (
                LicenseMetadata::Missing("package has no license or license-file metadata".into()),
                false,
                manifest_evidence,
            )
        };
        records.push(PackageRecord::from_lock(
            package,
            direct,
            license_metadata,
            rider,
            evidence,
        ));
    }
    records.sort_by(|left, right| (&left.name, &left.version).cmp(&(&right.name, &right.version)));
    records
}

fn inventory_json(audit: &Audit) -> String {
    let direct = audit
        .packages
        .iter()
        .filter(|package| package.direct)
        .count();
    let packages = audit
        .packages
        .iter()
        .map(|package| {
            let (license, license_file, heading, issue) = match &package.metadata {
                LicenseMetadata::Spdx(value) => (Some(value.as_str()), None, None, None),
                LicenseMetadata::File { path, heading } => {
                    (None, Some(path.as_str()), Some(heading.as_str()), None)
                }
                LicenseMetadata::Missing(reason) => (None, None, None, Some(reason.as_str())),
            };
            json!({
                "name": package.name,
                "version": package.version,
                "relationship": if package.direct { "direct" } else { "transitive" },
                "source": package.source,
                "checksum": package.checksum,
                "license": license,
                "license_file": license_file,
                "license_heading": heading,
                "rider": package.rider,
                "issue": issue,
                "evidence": package_evidence(package),
                "blocking_reasons": package_blocking_reasons(package),
            })
        })
        .collect::<Vec<_>>();
    format!(
        "{}\n",
        serde_json::to_string_pretty(&json!({
            "source": "Cargo.lock and Cargo registry manifests",
            "summary": {
                "packages": audit.packages.len(),
                "direct": direct,
                "transitive": audit.packages.len() - direct,
                "metadata_issues": audit.metadata_issues,
                "rider_packages": audit.rider_packages,
                "review_required": audit.review_required,
            },
            "packages": packages,
        }))
        .expect("license inventory values are always JSON serializable")
    )
}

fn run(config: &Config) -> Result<Audit, String> {
    let lock_path = config.workspace.join("Cargo.lock");
    let lock_content = fs::read_to_string(&lock_path)
        .map_err(|error| format!("failed to read {}: {error}", lock_path.display()))?;
    let packages = parse_lock(&lock_content)?;
    let direct_names = collect_direct_names(&config.workspace)?;
    let roots = registry_roots(config.registry_root.as_deref())?;
    let records = inspect_packages(packages, &direct_names, &roots);
    let audit = audit_packages(&records);
    validate_kernel_dependency_boundary(&audit)?;

    if let Some(path) = &config.inventory {
        let path = if path.is_absolute() {
            path.clone()
        } else {
            config.workspace.join(path)
        };
        fs::write(&path, inventory_json(&audit))
            .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    }
    Ok(audit)
}

fn usage() -> String {
    "Usage: focus-release-compliance [--mode technical|production] [--workspace PATH] [--registry-root PATH] [--inventory PATH]".into()
}

fn main() -> ExitCode {
    let config = match Config::parse(env::args().skip(1)) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let audit = match run(&config) {
        Ok(audit) => audit,
        Err(error) => {
            eprintln!("release compliance gate error: {error}");
            return ExitCode::FAILURE;
        }
    };
    let direct = audit
        .packages
        .iter()
        .filter(|package| package.direct)
        .count();
    println!(
        "packages={} direct={} transitive={} metadata_issues={} rider_packages={} review_required={}",
        audit.packages.len(),
        direct,
        audit.packages.len() - direct,
        audit.metadata_issues,
        audit.rider_packages,
        audit.review_required
    );
    if config.mode.allows(&audit) {
        println!("status=technical-build-allowed");
        ExitCode::SUCCESS
    } else {
        for blocker in production_blockers(&audit) {
            eprintln!(
                "blocker={}@{} reason={} evidence={}",
                blocker.package, blocker.version, blocker.reason, blocker.evidence
            );
        }
        eprintln!("status=production-distribution-review-required");
        ExitCode::from(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn technical_mode_allows_rider_but_marks_review_required() {
        let audit = audit_packages(&[
            PackageRecord::licensed("serde", "1.0.0", "MIT OR Apache-2.0"),
            PackageRecord::license_file(
                "pi_agent_rust",
                "0.1.23",
                "MIT License (with OpenAI/Anthropic Rider)",
            ),
        ]);

        assert!(audit.review_required);
        assert!(GateMode::Technical.allows(&audit));
        assert!(!GateMode::Production.allows(&audit));
    }

    #[test]
    fn production_boundary_accepts_complete_metadata_without_legacy_pi_dependency() {
        let audit = audit_packages(&[
            PackageRecord::licensed("focus-kernel", "0.1.0", "MIT OR Apache-2.0"),
            PackageRecord::licensed("serde", "1.0.0", "MIT OR Apache-2.0"),
        ]);

        assert!(validate_kernel_dependency_boundary(&audit).is_ok());
    }

    #[test]
    fn technical_boundary_allows_review_required_dependency_metadata() {
        let audit = audit_packages(&[PackageRecord::missing("unknown", "1.0.0")]);

        assert!(audit.review_required);
        assert!(validate_kernel_dependency_boundary(&audit).is_ok());
    }

    #[test]
    fn missing_license_metadata_blocks_production() {
        let audit = audit_packages(&[PackageRecord::missing("unknown", "1.0.0")]);

        assert!(audit.review_required);
        assert!(!GateMode::Production.allows(&audit));
    }

    #[test]
    fn rider_license_refs_and_non_spdx_slashes_require_review() {
        let audit = audit_packages(&[
            PackageRecord::licensed(
                "rider-ref",
                "1.0.0",
                "LicenseRef-MIT-OpenAI-Anthropic-Rider",
            ),
            PackageRecord::licensed("legacy", "1.0.0", "MIT/Apache-2.0"),
        ]);

        assert!(audit.review_required);
        assert_eq!(audit.rider_packages, 1);
        assert!(!GateMode::Production.allows(&audit));
    }

    #[test]
    fn normalizes_known_legacy_slash_spdx_expressions() {
        assert_eq!(
            normalize_legacy_slash_spdx("MIT/Apache-2.0").as_deref(),
            Some("MIT OR Apache-2.0")
        );
        assert_eq!(
            normalize_legacy_slash_spdx("Apache-2.0 / MIT").as_deref(),
            Some("Apache-2.0 OR MIT")
        );
        assert_eq!(normalize_legacy_slash_spdx("MIT/Unknown"), None);
    }

    #[test]
    fn production_blockers_name_packages_reasons_and_evidence() {
        let audit = audit_packages(&[
            PackageRecord::missing("uncached", "1.0.0"),
            PackageRecord::license_file(
                "rider-package",
                "2.0.0",
                "MIT License (with OpenAI/Anthropic Rider)",
            ),
        ]);

        let blockers = production_blockers(&audit);

        assert!(blockers.iter().any(|blocker| {
            blocker.package == "uncached"
                && blocker.reason.contains("missing metadata")
                && blocker.evidence.contains("test")
        }));
        assert!(blockers.iter().any(|blocker| {
            blocker.package == "rider-package"
                && blocker.reason.contains("Rider")
                && blocker.evidence.contains("test")
        }));
    }

    #[test]
    fn parses_lock_package_identity_and_checksum() {
        let packages = parse_lock(
            r#"
version = 4
[[package]]
name = "demo"
version = "1.2.3"
source = "registry+https://example.invalid/index"
checksum = "abc"
"#,
        )
        .unwrap();

        assert_eq!(packages.len(), 1);
        assert_eq!(packages[0].name, "demo");
        assert_eq!(packages[0].checksum.as_deref(), Some("abc"));
    }

    #[test]
    fn package_record_retains_lock_provenance() {
        let record = PackageRecord::from_lock(
            LockPackage {
                name: "demo".into(),
                version: "1.2.3".into(),
                source: Some("registry+https://example.invalid/index".into()),
                checksum: Some("abc".into()),
            },
            true,
            LicenseMetadata::Missing("registry source is not cached".into()),
            false,
            "registry source cache is missing `demo-1.2.3`".into(),
        );

        assert_eq!(record.name, "demo");
        assert_eq!(record.version, "1.2.3");
        assert!(record.direct);
        assert_eq!(
            record.source.as_deref(),
            Some("registry+https://example.invalid/index")
        );
        assert_eq!(record.checksum.as_deref(), Some("abc"));
    }

    #[test]
    fn reads_only_package_license_metadata() {
        let metadata = parse_manifest_license(
            r#"
[package]
license = "MIT OR Apache-2.0"
[dependencies]
license = "not-package-metadata"
"#,
        );

        assert_eq!(metadata.license.as_deref(), Some("MIT OR Apache-2.0"));
    }

    #[test]
    fn inventory_is_valid_json_shape_and_retains_relationship() {
        let mut package = PackageRecord::licensed("serde", "1.0.0", "MIT");
        package.direct = true;
        let inventory: serde_json::Value =
            serde_json::from_str(&inventory_json(&audit_packages(&[package]))).unwrap();

        assert_eq!(inventory["packages"][0]["relationship"], "direct");
        assert_eq!(inventory["packages"][0]["license"], "MIT");
        assert!(
            inventory["packages"][0]["evidence"]
                .as_str()
                .is_some_and(|evidence| evidence.starts_with("test package metadata"))
        );
        assert_eq!(inventory["packages"][0]["blocking_reasons"], json!([]));
    }
}
