//! Bounded outbound HTTP policy shared by the Runtime-owned web tool.

use std::{
    collections::BTreeMap,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    time::Duration,
};

use focus_kernel::CancellationSignal;
use reqwest::Url;
use serde::{Deserialize, Serialize};

use crate::{RuntimeError, cancellation::wait_for_cancellation};

/// Per-domain outcome for the Runtime web client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DomainAccess {
    /// Permit a matching destination.
    Allow,
    /// Block a matching destination. Deny rules always win.
    Deny,
}

/// Explicit configuration for bounded outbound web fetches.
///
/// With no rules, enabled networking permits public destinations. Adding one
/// allow rule switches external destinations to allowlist behavior; deny-only
/// rules remain a public-network blocklist. Local and private destinations
/// remain blocked unless explicitly configured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Whether the Runtime advertises its first-party web fetch tool.
    pub enabled: bool,
    /// Exact and wildcard domain rules. `deny` always takes precedence.
    pub domains: BTreeMap<String, DomainAccess>,
    /// Permit local and private addresses after domain-rule validation.
    pub allow_local: bool,
    /// Explicit destination ports. Empty uses HTTP(S) defaults only.
    #[serde(default)]
    pub allowed_ports: Vec<u16>,
    /// Maximum number of redirects followed by one request.
    pub max_redirects: usize,
    /// Maximum retained response bytes per fetch.
    pub max_response_bytes: usize,
    /// Whole-request deadline for one HTTP hop.
    #[serde(with = "duration_millis")]
    pub request_timeout: Duration,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            domains: BTreeMap::new(),
            allow_local: false,
            allowed_ports: Vec::new(),
            max_redirects: 5,
            max_response_bytes: 1_000_000,
            request_timeout: Duration::from_secs(30),
        }
    }
}

impl NetworkConfig {
    /// Build a public-network configuration with bounded defaults.
    #[must_use]
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Add one validated exact host, `*.subdomain`, `**.domain`, or `*` rule.
    pub fn insert_domain_rule(
        &mut self,
        pattern: impl AsRef<str>,
        access: DomainAccess,
    ) -> Result<(), RuntimeError> {
        let pattern = normalize_pattern(pattern.as_ref())?;
        self.domains.insert(pattern, access);
        Ok(())
    }

    /// Validate public configuration before the Runtime opens network tooling.
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.enabled && self.max_redirects > 10 {
            return Err(RuntimeError::Network(
                "network redirect limit must be at most 10".into(),
            ));
        }
        if self.enabled && !(1..=8 * 1024 * 1024).contains(&self.max_response_bytes) {
            return Err(RuntimeError::Network(
                "network response limit must be between 1 and 8388608 bytes".into(),
            ));
        }
        if self.enabled && self.request_timeout.is_zero() {
            return Err(RuntimeError::Network(
                "network request timeout must be positive".into(),
            ));
        }
        if self.allowed_ports.contains(&0) {
            return Err(RuntimeError::Network(
                "network ports must be between 1 and 65535".into(),
            ));
        }
        for pattern in self.domains.keys() {
            let normalized = normalize_pattern(pattern)?;
            if normalized != *pattern {
                return Err(RuntimeError::Network(format!(
                    "domain pattern `{pattern}` must be normalized"
                )));
            }
        }
        Ok(())
    }

    /// Permit one non-standard HTTP(S) port after configuration validation.
    pub fn allow_port(&mut self, port: u16) {
        if !self.allowed_ports.contains(&port) {
            self.allowed_ports.push(port);
            self.allowed_ports.sort_unstable();
        }
    }

    /// Check a URL before every request and redirect hop.
    pub async fn authorize_url(
        &self,
        url: &Url,
        cancellation: &dyn CancellationSignal,
    ) -> Result<Vec<SocketAddr>, RuntimeError> {
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        if !self.enabled {
            return Err(RuntimeError::Network("networking is disabled".into()));
        }
        if !matches!(url.scheme(), "http" | "https") {
            return Err(RuntimeError::Network(format!(
                "unsupported URL scheme `{}`",
                url.scheme()
            )));
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(RuntimeError::Network(
                "URLs with embedded credentials are not supported".into(),
            ));
        }
        let host = url
            .host_str()
            .ok_or_else(|| RuntimeError::Network("URL must include a host".into()))?
            .trim_end_matches('.')
            .to_ascii_lowercase();
        if host.is_empty() {
            return Err(RuntimeError::Network("URL must include a host".into()));
        }
        if !self.domain_allowed(&host) {
            return Err(RuntimeError::Network(format!(
                "destination `{host}` is not permitted by the network policy"
            )));
        }

        let port = url.port_or_known_default().ok_or_else(|| {
            RuntimeError::Network(format!("URL has no known port for `{}`", url.scheme()))
        })?;
        let default_port = match url.scheme() {
            "http" => 80,
            "https" => 443,
            _ => unreachable!("scheme was validated above"),
        };
        if port != default_port && !self.allowed_ports.contains(&port) {
            return Err(RuntimeError::Network(format!(
                "destination port `{port}` is not permitted by the network policy"
            )));
        }
        let addresses = tokio::select! {
            () = wait_for_cancellation(cancellation) => return Err(RuntimeError::Cancelled),
            resolved = tokio::time::timeout(
                self.request_timeout,
                tokio::net::lookup_host((host.as_str(), port)),
            ) => resolved
                .map_err(|_| RuntimeError::Network(format!(
                    "DNS lookup timed out for `{host}` after {:?}",
                    self.request_timeout
                )))?
                .map_err(|error| RuntimeError::Network(format!(
                    "DNS lookup failed for `{host}`: {error}"
                )))?,
        };
        let addresses = addresses.collect::<Vec<_>>();
        if addresses.is_empty() {
            return Err(RuntimeError::Network(format!(
                "DNS lookup returned no addresses for `{host}`"
            )));
        }
        if !self.allow_local && addresses.iter().any(|address| is_local(address.ip())) {
            return Err(RuntimeError::Network(format!(
                "destination `{host}` resolved to a local or private address"
            )));
        }
        Ok(addresses)
    }

    fn domain_allowed(&self, host: &str) -> bool {
        let mut allowed = !self
            .domains
            .values()
            .any(|access| *access == DomainAccess::Allow);
        for (pattern, access) in &self.domains {
            if domain_matches(pattern, host) {
                match access {
                    DomainAccess::Deny => return false,
                    DomainAccess::Allow => allowed = true,
                }
            }
        }
        allowed
    }
}

fn normalize_pattern(pattern: &str) -> Result<String, RuntimeError> {
    let raw = pattern.trim();
    if raw.ends_with('.') {
        return Err(RuntimeError::Network(format!(
            "invalid domain pattern `{raw}`"
        )));
    }
    let pattern = raw.to_ascii_lowercase();
    if pattern.is_empty()
        || pattern.contains(['/', ':', '@', '?', '#'])
        || pattern.split('.').any(|part| part.is_empty())
    {
        return Err(RuntimeError::Network(format!(
            "invalid domain pattern `{pattern}`"
        )));
    }
    let valid = pattern == "*"
        || pattern
            .strip_prefix("*.")
            .or_else(|| pattern.strip_prefix("**."))
            .is_some_and(is_valid_host_pattern)
        || is_valid_host_pattern(&pattern);
    if valid {
        Ok(pattern)
    } else {
        Err(RuntimeError::Network(format!(
            "invalid domain pattern `{pattern}`"
        )))
    }
}

fn is_valid_host_pattern(host: &str) -> bool {
    !host.is_empty()
        && host
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
}

fn domain_matches(pattern: &str, host: &str) -> bool {
    match pattern {
        "*" => true,
        pattern if pattern.starts_with("**.") => {
            let suffix = &pattern[3..];
            host == suffix
                || host
                    .strip_suffix(suffix)
                    .is_some_and(|prefix| prefix.ends_with('.'))
        }
        pattern if pattern.starts_with("*.") => {
            let suffix = &pattern[2..];
            host != suffix
                && host
                    .strip_suffix(suffix)
                    .is_some_and(|prefix| prefix.ends_with('.'))
        }
        pattern => host == pattern,
    }
}

fn is_local(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_local_v4(address),
        IpAddr::V6(address) => is_local_v6(address),
    }
}

fn is_local_v4(address: Ipv4Addr) -> bool {
    let octets = address.octets();
    let documentation = matches!(
        octets,
        [192, 0, 2, _] | [198, 51, 100, _] | [203, 0, 113, _]
    );
    let shared = octets[0] == 100 && (64..=127).contains(&octets[1]);
    let benchmarking = octets[0] == 198 && matches!(octets[1], 18 | 19);
    address.is_private()
        || address.is_loopback()
        || address.is_link_local()
        || address.is_broadcast()
        || address.is_unspecified()
        || address.is_multicast()
        || octets[0] >= 240
        || shared
        || benchmarking
        || documentation
}

fn is_local_v6(address: Ipv6Addr) -> bool {
    if let Some(mapped) = address.to_ipv4_mapped() {
        return is_local_v4(mapped);
    }
    let segments = address.segments();
    let documentation = segments[0] == 0x2001 && segments[1] == 0x0db8;
    address.is_loopback()
        || address.is_unspecified()
        || address.is_unique_local()
        || address.is_unicast_link_local()
        || address.is_multicast()
        || documentation
}

mod duration_millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis().min(u128::from(u64::MAX)) as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        u64::deserialize(deserializer).map(Duration::from_millis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_rules_support_exact_and_scoped_wildcards_with_deny_precedence() {
        let mut config = NetworkConfig::enabled();
        config
            .insert_domain_rule("*.example.com", DomainAccess::Allow)
            .unwrap();
        config
            .insert_domain_rule("api.example.com", DomainAccess::Deny)
            .unwrap();

        assert!(config.domain_allowed("docs.example.com"));
        assert!(!config.domain_allowed("example.com"));
        assert!(!config.domain_allowed("api.example.com"));
        assert!(!config.domain_allowed("other.example.net"));
    }

    #[test]
    fn double_star_matches_apex_and_subdomains() {
        let mut config = NetworkConfig::enabled();
        config
            .insert_domain_rule("**.example.com", DomainAccess::Allow)
            .unwrap();

        assert!(config.domain_allowed("example.com"));
        assert!(config.domain_allowed("api.example.com"));
    }

    #[test]
    fn invalid_domain_rules_fail_before_runtime_start() {
        let mut config = NetworkConfig::enabled();
        assert!(
            config
                .insert_domain_rule("https://example.com", DomainAccess::Allow)
                .is_err()
        );
        assert!(
            config
                .insert_domain_rule("*.", DomainAccess::Allow)
                .is_err()
        );
    }

    #[test]
    fn enabled_configuration_rejects_zero_response_limit() {
        let mut config = NetworkConfig::enabled();
        config.max_response_bytes = 0;

        assert!(config.validate().is_err());
    }

    #[test]
    fn enabled_configuration_rejects_excessive_response_limit() {
        let mut config = NetworkConfig::enabled();
        config.max_response_bytes = 8 * 1024 * 1024 + 1;

        assert!(config.validate().is_err());
    }

    #[test]
    fn enabled_configuration_rejects_unbounded_redirects() {
        let mut config = NetworkConfig::enabled();
        config.max_redirects = 11;

        assert!(config.validate().is_err());
    }

    #[test]
    fn legacy_serialized_configuration_defaults_to_standard_ports() {
        let config: NetworkConfig = serde_json::from_str(
            r#"{"enabled":true,"domains":{},"allow_local":false,"max_redirects":5,"max_response_bytes":100,"request_timeout":1000}"#,
        )
        .unwrap();

        assert!(config.allowed_ports.is_empty());
        config.validate().unwrap();
    }

    #[tokio::test]
    async fn non_standard_ports_require_explicit_permission() {
        let mut config = NetworkConfig::enabled();
        config.allow_local = true;
        let url = Url::parse("http://127.0.0.1:8080/").unwrap();

        assert!(
            config
                .authorize_url(&url, &focus_kernel::NoCancellation)
                .await
                .is_err()
        );
        config.allow_port(8080);
        config
            .authorize_url(&url, &focus_kernel::NoCancellation)
            .await
            .unwrap();
    }

    #[test]
    fn non_public_shared_multicast_and_mapped_private_addresses_are_blocked() {
        assert!(is_local("100.64.0.1".parse().unwrap()));
        assert!(is_local("198.18.0.1".parse().unwrap()));
        assert!(is_local("224.0.0.1".parse().unwrap()));
        assert!(is_local("::ffff:192.168.1.1".parse().unwrap()));
        assert!(is_local("ff02::1".parse().unwrap()));
    }

    #[tokio::test]
    async fn local_destinations_need_an_explicit_configuration() {
        let mut config = NetworkConfig::enabled();
        config
            .insert_domain_rule("localhost", DomainAccess::Allow)
            .unwrap();
        let local = Url::parse("http://localhost:12345/").unwrap();

        assert!(
            config
                .authorize_url(&local, &focus_kernel::NoCancellation)
                .await
                .is_err()
        );
        config.allow_local = true;
        config.allow_port(12_345);
        config
            .authorize_url(&local, &focus_kernel::NoCancellation)
            .await
            .unwrap();
        config.allow_local = false;
        config.domains.clear();
        assert!(
            config
                .authorize_url(&local, &focus_kernel::NoCancellation)
                .await
                .is_err()
        );
    }
}
