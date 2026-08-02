use serde::{Deserialize, Deserializer, Serialize, de::IgnoredAny};

fn deserialize_internal_false<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    IgnoredAny::deserialize(deserializer)?;
    Ok(false)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(
        default,
        skip_serializing,
        deserialize_with = "deserialize_internal_false"
    )]
    pub bypass_provider_model_override: bool,
    #[serde(
        default,
        skip_serializing,
        deserialize_with = "deserialize_internal_false"
    )]
    pub bypass_provider_effort_override: bool,
    #[serde(
        default,
        skip_serializing,
        deserialize_with = "deserialize_internal_false"
    )]
    pub auxiliary_request: bool,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountTokensResponse {
    pub input_tokens: u64,
}
