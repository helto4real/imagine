//! Only this machine may address the token-free local API. Loopback binding keeps
//! other hosts away, but a web page whose domain rebinds to 127.0.0.1 reaches the
//! port from a browser on this machine: its Host names that domain and its Origin
//! that page. Requests without a Host or an Origin are local clients (native
//! clients and in-process callers); browsers always send a Host.

fn loopback(name: &str) -> bool {
    matches!(name, "127.0.0.1" | "localhost" | "[::1]" | "::1")
}
/// The host name of `host:port` or `[v6]:port`.
fn host_name(authority: &str) -> &str {
    if authority.starts_with('[') {
        return authority
            .find(']')
            .map_or(authority, |end| &authority[..=end]);
    }
    authority
        .rsplit_once(':')
        .map_or(authority, |(name, _)| name)
}
/// `Err` with the response code when Host or Origin names another site.
pub fn check(host: Option<&str>, origin: Option<&str>) -> Result<(), &'static str> {
    if host.is_some_and(|h| !loopback(&host_name(h).to_ascii_lowercase())) {
        return Err("invalid_host");
    }
    if let Some(origin) = origin {
        let authority = origin
            .strip_prefix("http://")
            .or_else(|| origin.strip_prefix("https://"))
            .filter(|a| !a.contains('/'));
        if !authority.is_some_and(|a| loopback(&host_name(a).to_ascii_lowercase())) {
            return Err("invalid_origin");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check;

    #[test]
    fn only_loopback_hosts_and_origins_pass() {
        for host in [
            None,
            Some("127.0.0.1:8080"),
            Some("localhost:8189"),
            Some("[::1]:8080"),
            Some("LOCALHOST"),
        ] {
            assert_eq!(check(host, None), Ok(()), "{host:?}");
        }
        for host in [
            "evil.example:8080",
            "127.0.0.1.evil.example:8080",
            "192.168.1.5:8080",
            "[::2]:8080",
        ] {
            assert_eq!(check(Some(host), None), Err("invalid_host"), "{host}");
        }
        for origin in [
            "http://127.0.0.1:8000",
            "http://localhost:8189",
            "https://[::1]:443",
        ] {
            assert_eq!(
                check(Some("127.0.0.1:8189"), Some(origin)),
                Ok(()),
                "{origin}"
            );
        }
        for origin in [
            "http://evil.example",
            "null",
            "http://localhost.evil.example:8189",
            "file://",
            "http://127.0.0.1:8000/x",
        ] {
            assert_eq!(
                check(Some("127.0.0.1:8189"), Some(origin)),
                Err("invalid_origin"),
                "{origin}"
            );
        }
    }
}
