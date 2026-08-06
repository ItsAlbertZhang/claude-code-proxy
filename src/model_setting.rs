use crate::anthropic::schema::MessagesRequest;
use anyhow::{Context, bail};
use http::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use url::Url;

pub const MODEL_SETTING_ENV: &str = "CCP_MODEL_SETTING";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ClaudeAliasFamily {
    Haiku,
    Sonnet,
    Opus,
    Fable,
}

impl ClaudeAliasFamily {
    pub fn from_normalized_model(model: &str) -> Option<Self> {
        match model {
            "haiku" | "claude-haiku-4-5" | "claude-haiku-4-5-20251001" => Some(Self::Haiku),
            "sonnet" | "claude-sonnet-4-6" | "claude-sonnet-5" => Some(Self::Sonnet),
            "opus" | "claude-opus-4-7" | "claude-opus-4-8" | "claude-opus-5" => Some(Self::Opus),
            "fable" | "claude-fable-5" => Some(Self::Fable),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Haiku => "haiku",
            Self::Sonnet => "sonnet",
            Self::Opus => "opus",
            Self::Fable => "fable",
        }
    }
}

#[derive(Clone)]
pub struct ModelSetting {
    routes: BTreeMap<ClaudeAliasFamily, ConfiguredAnthropicRoute>,
}

#[derive(Clone)]
pub struct ConfiguredAnthropicRoute {
    url: Url,
    authorization: HeaderValue,
    headers: HeaderMap,
    model: String,
    default_effort: Option<String>,
    effort_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelSetting {
    version: u32,
    #[serde(default)]
    routes: RawRoutes,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRoutes {
    haiku: Option<RawRoute>,
    sonnet: Option<RawRoute>,
    opus: Option<RawRoute>,
    fable: Option<RawRoute>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawRoute {
    url: String,
    api_key: String,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    model: String,
    default_effort: Option<String>,
    #[serde(default)]
    effort_map: BTreeMap<String, String>,
}

impl ModelSetting {
    pub fn load_from_env() -> anyhow::Result<Option<Self>> {
        let Some(raw_path) = std::env::var_os(MODEL_SETTING_ENV) else {
            return Ok(None);
        };
        if raw_path.to_string_lossy().trim().is_empty() {
            return Ok(None);
        }
        Self::load_from_path(PathBuf::from(raw_path)).map(Some)
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path).with_context(|| {
            format!("failed to read {MODEL_SETTING_ENV} file {}", path.display())
        })?;
        Self::parse(&raw)
            .with_context(|| format!("failed to load {MODEL_SETTING_ENV} file {}", path.display()))
    }

    pub(crate) fn parse(raw: &str) -> anyhow::Result<Self> {
        let raw: RawModelSetting = serde_json::from_str(raw)
            .context("model setting must be valid JSON matching version 1 schema")?;
        if raw.version != 1 {
            bail!(
                "unsupported model setting version {}; expected 1",
                raw.version
            );
        }

        let mut routes = BTreeMap::new();
        for (family, route) in [
            (ClaudeAliasFamily::Haiku, raw.routes.haiku),
            (ClaudeAliasFamily::Sonnet, raw.routes.sonnet),
            (ClaudeAliasFamily::Opus, raw.routes.opus),
            (ClaudeAliasFamily::Fable, raw.routes.fable),
        ] {
            if let Some(route) = route {
                routes.insert(family, ConfiguredAnthropicRoute::validate(family, route)?);
            }
        }
        Ok(Self { routes })
    }

    pub(crate) fn into_routes(
        self,
    ) -> impl Iterator<Item = (ClaudeAliasFamily, ConfiguredAnthropicRoute)> {
        self.routes.into_iter()
    }

    #[cfg(test)]
    fn route(&self, family: ClaudeAliasFamily) -> Option<&ConfiguredAnthropicRoute> {
        self.routes.get(&family)
    }
}

impl ConfiguredAnthropicRoute {
    fn validate(family: ClaudeAliasFamily, raw: RawRoute) -> anyhow::Result<Self> {
        let route_name = family.as_str();
        if raw.api_key.is_empty() {
            bail!("routes.{route_name}.apiKey must not be empty");
        }
        if raw.model.trim().is_empty() {
            bail!("routes.{route_name}.model must not be empty");
        }

        let url = Url::parse(&raw.url)
            .with_context(|| format!("routes.{route_name}.url must be a valid URL"))?;
        if !matches!(url.scheme(), "http" | "https") {
            bail!("routes.{route_name}.url must use http or https");
        }
        if !url.username().is_empty() || url.password().is_some() {
            bail!("routes.{route_name}.url must not contain credentials");
        }
        if url.fragment().is_some() {
            bail!("routes.{route_name}.url must not contain a fragment");
        }

        let mut headers = HeaderMap::new();
        for (name, value) in raw.headers {
            let parsed_name = HeaderName::from_bytes(name.as_bytes())
                .with_context(|| format!("routes.{route_name}.headers contains invalid name"))?;
            if is_reserved_header(&parsed_name) {
                bail!(
                    "routes.{route_name}.headers must not set reserved header {}",
                    parsed_name.as_str()
                );
            }
            let parsed_value = HeaderValue::from_str(&value).with_context(|| {
                format!(
                    "routes.{route_name}.headers.{} contains an invalid value",
                    parsed_name.as_str()
                )
            })?;
            headers.insert(parsed_name, parsed_value);
        }

        if raw.default_effort.as_deref().is_some_and(str::is_empty) {
            bail!("routes.{route_name}.defaultEffort must not be empty");
        }
        for (source, target) in &raw.effort_map {
            if source.is_empty() || target.is_empty() {
                bail!("routes.{route_name}.effortMap keys and values must not be empty");
            }
        }

        let mut authorization = HeaderValue::from_str(&format!("Bearer {}", raw.api_key))
            .with_context(|| format!("routes.{route_name}.apiKey is not a valid bearer token"))?;
        authorization.set_sensitive(true);

        Ok(Self {
            url,
            authorization,
            headers,
            model: raw.model,
            default_effort: raw.default_effort,
            effort_map: raw.effort_map,
        })
    }

    pub(crate) fn url(&self) -> &Url {
        &self.url
    }

    pub(crate) fn authorization(&self) -> &HeaderValue {
        &self.authorization
    }

    pub(crate) fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    pub(crate) fn prepare_body(&self, body: &MessagesRequest) -> serde_json::Result<Value> {
        let mut value = serde_json::to_value(body)?;
        value["model"] = Value::String(self.model.clone());
        let configured_effort = match value.pointer("/output_config/effort") {
            Some(Value::String(effort)) => self.effort_map.get(effort).cloned(),
            Some(_) => None,
            None => self.default_effort.clone(),
        };
        if let Some(effort) = configured_effort {
            if let Some(current) = value.pointer_mut("/output_config/effort") {
                *current = Value::String(effort);
            } else if let Some(root) = value.as_object_mut() {
                let output_config = root
                    .entry("output_config")
                    .or_insert_with(|| Value::Object(Default::default()));
                if let Some(output_config) = output_config.as_object_mut() {
                    output_config.insert("effort".to_string(), Value::String(effort));
                }
            }
        }
        Ok(value)
    }
}

fn is_reserved_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "accept"
            | "authorization"
            | "connection"
            | "content-length"
            | "content-type"
            | "host"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn route_json(extra: Value) -> String {
        let mut route = json!({
            "url": "http://127.0.0.1:12345/v1/messages?beta=true",
            "apiKey": "secret-value",
            "headers": {"x-client-type": "example"},
            "model": "target-model",
            "effortMap": {"max": "xhigh"}
        });
        if let (Some(route), Some(extra)) = (route.as_object_mut(), extra.as_object()) {
            route.extend(extra.clone());
        }
        json!({"version": 1, "routes": {"sonnet": route}}).to_string()
    }

    #[test]
    fn parses_only_configured_alias_families() {
        let setting = ModelSetting::parse(
            &json!({
                "version": 1,
                "routes": {
                    "sonnet": {
                        "url": "http://127.0.0.1:1/v1/messages",
                        "apiKey": "one",
                        "model": "sonnet-target"
                    },
                    "fable": {
                        "url": "http://127.0.0.1:2/v1/messages",
                        "apiKey": "two",
                        "model": "fable-target"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();

        assert!(setting.route(ClaudeAliasFamily::Haiku).is_none());
        assert!(setting.route(ClaudeAliasFamily::Opus).is_none());
        assert_eq!(
            setting.route(ClaudeAliasFamily::Sonnet).unwrap().model(),
            "sonnet-target"
        );
        assert_eq!(
            setting.route(ClaudeAliasFamily::Fable).unwrap().model(),
            "fable-target"
        );
    }

    #[test]
    fn accepts_each_supported_alias_family() {
        let setting = ModelSetting::parse(
            &json!({
                "version": 1,
                "routes": {
                    "haiku": {
                        "url": "http://127.0.0.1:1/v1/messages",
                        "apiKey": "one",
                        "model": "haiku-target"
                    },
                    "sonnet": {
                        "url": "http://127.0.0.1:2/v1/messages",
                        "apiKey": "two",
                        "model": "sonnet-target"
                    },
                    "opus": {
                        "url": "http://127.0.0.1:3/v1/messages",
                        "apiKey": "three",
                        "model": "opus-target"
                    },
                    "fable": {
                        "url": "http://127.0.0.1:4/v1/messages",
                        "apiKey": "four",
                        "model": "fable-target"
                    }
                }
            })
            .to_string(),
        )
        .unwrap();

        for family in [
            ClaudeAliasFamily::Haiku,
            ClaudeAliasFamily::Sonnet,
            ClaudeAliasFamily::Opus,
            ClaudeAliasFamily::Fable,
        ] {
            assert!(
                setting.route(family).is_some(),
                "missing {}",
                family.as_str()
            );
        }
    }

    #[test]
    fn loads_setting_from_explicit_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("routes.json");
        std::fs::write(&path, route_json(json!({}))).unwrap();

        let setting = ModelSetting::load_from_path(path).unwrap();

        assert!(setting.route(ClaudeAliasFamily::Sonnet).is_some());
        assert!(setting.route(ClaudeAliasFamily::Opus).is_none());
    }

    #[test]
    fn rejects_unknown_route_keys() {
        let error = ModelSetting::parse(
            &json!({
                "version": 1,
                "routes": {
                    "custom": {
                        "url": "http://127.0.0.1/v1/messages",
                        "apiKey": "secret-value",
                        "model": "target"
                    }
                }
            })
            .to_string(),
        )
        .err()
        .expect("unknown route should fail")
        .to_string();
        assert!(!error.contains("secret-value"));
    }

    #[test]
    fn rejects_reserved_headers_without_exposing_values() {
        let error = ModelSetting::parse(&route_json(json!({
            "headers": {"authorization": "another-secret"}
        })))
        .err()
        .expect("reserved header should fail")
        .to_string();
        assert!(error.contains("reserved header authorization"));
        assert!(!error.contains("another-secret"));
        assert!(!error.contains("secret-value"));
    }

    #[test]
    fn maps_model_and_configured_effort_only() {
        let setting = ModelSetting::parse(&route_json(json!({}))).unwrap();
        let route = setting.route(ClaudeAliasFamily::Sonnet).unwrap();
        let body: MessagesRequest = serde_json::from_value(json!({
            "model": "sonnet",
            "max_tokens": 16,
            "messages": [],
            "stream": true,
            "output_config": {"effort": "max", "other": true}
        }))
        .unwrap();
        let prepared = route.prepare_body(&body).unwrap();

        assert_eq!(prepared["model"], "target-model");
        assert_eq!(prepared["output_config"]["effort"], "xhigh");
        assert_eq!(prepared["output_config"]["other"], true);
    }

    #[test]
    fn applies_configured_default_effort_when_downstream_omits_it() {
        let setting = ModelSetting::parse(&route_json(json!({
            "defaultEffort": "xhigh"
        })))
        .unwrap();
        let route = setting.route(ClaudeAliasFamily::Sonnet).unwrap();
        let body: MessagesRequest = serde_json::from_value(json!({
            "model": "sonnet",
            "max_tokens": 16,
            "messages": [],
            "stream": true,
            "output_config": {"format": {"type": "json_schema"}}
        }))
        .unwrap();

        let prepared = route.prepare_body(&body).unwrap();

        assert_eq!(prepared["output_config"]["effort"], "xhigh");
        assert_eq!(
            prepared["output_config"]["format"],
            json!({"type": "json_schema"})
        );
    }

    #[test]
    fn leaves_omitted_effort_absent_without_a_configured_default() {
        let setting = ModelSetting::parse(&route_json(json!({}))).unwrap();
        let route = setting.route(ClaudeAliasFamily::Sonnet).unwrap();
        let body: MessagesRequest = serde_json::from_value(json!({
            "model": "sonnet",
            "max_tokens": 16,
            "messages": [],
            "stream": true
        }))
        .unwrap();

        let prepared = route.prepare_body(&body).unwrap();

        assert!(prepared.pointer("/output_config/effort").is_none());
    }

    #[test]
    fn recognizes_only_existing_aliases() {
        for (model, family) in [
            ("haiku", ClaudeAliasFamily::Haiku),
            ("claude-haiku-4-5-20251001", ClaudeAliasFamily::Haiku),
            ("sonnet", ClaudeAliasFamily::Sonnet),
            ("claude-sonnet-5", ClaudeAliasFamily::Sonnet),
            ("opus", ClaudeAliasFamily::Opus),
            ("claude-opus-4-8", ClaudeAliasFamily::Opus),
            ("fable", ClaudeAliasFamily::Fable),
            ("claude-fable-5", ClaudeAliasFamily::Fable),
        ] {
            assert_eq!(
                ClaudeAliasFamily::from_normalized_model(model),
                Some(family)
            );
        }
        assert_eq!(
            ClaudeAliasFamily::from_normalized_model("claude-sonnet-future"),
            None
        );
        assert_eq!(ClaudeAliasFamily::from_normalized_model("glm-5.2"), None);
    }
}
