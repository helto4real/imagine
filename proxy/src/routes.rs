use fancy_regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;

static POLICY: LazyLock<Value> =
    LazyLock::new(|| serde_json::from_str(include_str!("../reference/comfy-routes.json")).unwrap());
static PATTERNS: LazyLock<Vec<(Vec<String>, Regex)>> = LazyLock::new(|| {
    POLICY["SAFE_CUSTOM_MUTATION_PATTERNS"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v["methods"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().into())
                    .collect(),
                Regex::new(v["pattern"].as_str().unwrap()).unwrap(),
            )
        })
        .collect()
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Passthrough,
    Workflow,
    Control,
    Lifecycle,
    BlockedGpu,
    UnknownMutation,
}
impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Passthrough => "passthrough",
            Self::Workflow => "workflow",
            Self::Control => "control",
            Self::Lifecycle => "lifecycle",
            Self::BlockedGpu => "blocked_gpu",
            Self::UnknownMutation => "unknown_mutation",
        }
    }
}
fn contains(key: &str, value: &str) -> bool {
    POLICY[key].as_array().unwrap().iter().any(|v| v == value)
}
fn route(key: &str, method: &str, path: &str) -> bool {
    POLICY[key]
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v[0] == method && v[1] == path)
}
pub fn upstream_path(path: &str) -> String {
    if let Some(alias) = POLICY["EXACT_UPSTREAM_ROUTE_ALIASES"][path].as_str() {
        return alias.into();
    }
    for (public, upstream) in POLICY["PREFIX_UPSTREAM_ROUTE_ALIASES"].as_object().unwrap() {
        if path == public || path.starts_with(&format!("{public}/")) {
            return format!("{}{}", upstream.as_str().unwrap(), &path[public.len()..]);
        }
    }
    path.into()
}
pub fn classify(method: &str, path: &str) -> Kind {
    let method = method.to_uppercase();
    let m = method.as_str();
    if matches!(m, "GET" | "HEAD" | "OPTIONS") {
        return Kind::Passthrough;
    }
    if route("WORKFLOW_ROUTES", m, path) {
        return Kind::Workflow;
    }
    let parts: Vec<_> = path.trim_matches('/').split('/').collect();
    let cancel = path == "/api/jobs/cancel"
        || (parts.len() == 4 && parts[..2] == ["api", "jobs"] && parts[3] == "cancel");
    if route("CONTROL_ROUTES", m, path) || (m == "POST" && cancel) {
        return Kind::Control;
    }
    if route("LIFECYCLE_ROUTES", m, path) {
        return Kind::Lifecycle;
    }
    if route("BLOCKED_GPU_ROUTES", m, path) {
        return Kind::BlockedGpu;
    }
    if contains("MUTATING_METHODS", m)
        && (contains("SAFE_CORE_MUTATION_PATHS", path)
            || POLICY["SAFE_CORE_MUTATION_PREFIXES"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| path.starts_with(p.as_str().unwrap())))
    {
        return Kind::Passthrough;
    }
    if (m == "POST" && contains("SAFE_CUSTOM_POST_PATHS", path))
        || route("SAFE_CUSTOM_MUTATION_ROUTES", m, path)
        || PATTERNS
            .iter()
            .any(|(methods, p)| methods.iter().any(|x| x == m) && p.is_match(path).unwrap_or(false))
    {
        return Kind::Passthrough;
    }
    Kind::UnknownMutation
}
