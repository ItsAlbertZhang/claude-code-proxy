use serde_json::Value;

/// Layer 1 keeps Read rewriting stateless. Stable, lane-scoped rewrite state is
/// intentionally deferred; callers therefore never receive a retained note.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadOffsetRewrite {
    pub offset: i64,
    pub file_path: Option<String>,
}

pub fn sanitize_read_args(name: &str, args: &str, _call_id: Option<&str>) -> String {
    if name != "Read" || args.is_empty() {
        return args.to_string();
    }

    let parsed: Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return args.to_string(),
    };
    let obj = match parsed.as_object() {
        Some(o) => o,
        None => return args.to_string(),
    };

    let mut sanitized = obj.clone();
    let mut changed = false;

    let has_empty_pages = obj
        .get("pages")
        .and_then(|v| v.as_str())
        .is_some_and(|s| s.is_empty());
    if has_empty_pages {
        sanitized.remove("pages");
        changed = true;
    }

    if changed {
        serde_json::to_string(&sanitized).unwrap_or_else(|_| args.to_string())
    } else {
        args.to_string()
    }
}

pub fn read_offset_rewrite(_call_id: &str) -> Option<ReadOffsetRewrite> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_read_args_removes_empty_pages() {
        let args = r#"{"file_path":"/tmp/a","pages":""}"#;
        let sanitized = sanitize_read_args("Read", args, None);
        let parsed: Value = serde_json::from_str(&sanitized).unwrap();
        assert!(parsed.get("pages").is_none());
        assert_eq!(
            parsed.get("file_path").and_then(|v| v.as_str()),
            Some("/tmp/a")
        );
    }

    #[test]
    fn read_offset_rewrite_is_disabled_without_a_stable_lane() {
        let args = r#"{"file_path":"/tmp/a","offset":1300000,"limit":20}"#;
        let sanitized = sanitize_read_args("Read", args, Some("call_rewrite_test"));
        let parsed: Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(
            parsed.get("offset").and_then(|v| v.as_i64()),
            Some(1_300_000)
        );
        assert_eq!(parsed.get("limit").and_then(|v| v.as_i64()), Some(20));
        assert!(read_offset_rewrite("call_rewrite_test").is_none());
    }

    #[test]
    fn sanitize_read_args_keeps_normal_offset() {
        let args = r#"{"file_path":"/tmp/a","offset":1300,"limit":20}"#;
        let sanitized = sanitize_read_args("Read", args, Some("call_keep_test"));
        let parsed: Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed.get("offset").and_then(|v| v.as_i64()), Some(1_300));
        assert!(read_offset_rewrite("call_keep_test").is_none());
    }
}
