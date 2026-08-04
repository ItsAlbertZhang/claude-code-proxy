use claude_code_proxy::openai_compat::OpenAiResponseMetadata;
use claude_code_proxy::providers::grok::translate::request::GrokResponsesRequest;
use claude_code_proxy::providers::kimi::translate::request::{KimiChatRequest, KimiStreamOptions};

#[test]
fn pre_m5_public_request_literals_compile() {
    let metadata = OpenAiResponseMetadata {
        tools: Vec::new(),
        tool_choice: serde_json::json!("auto"),
    };
    assert!(metadata.tools.is_empty());

    let grok = GrokResponsesRequest {
        model: "grok-4.5".into(),
        instructions: None,
        input: Vec::new(),
        tools: None,
        tool_choice: None,
        store: false,
        stream: true,
        max_output_tokens: None,
    };
    assert_eq!(grok.model, "grok-4.5");

    let kimi = KimiChatRequest {
        model: "kimi-for-coding".into(),
        messages: Vec::new(),
        tools: None,
        tool_choice: None,
        stream: true,
        stream_options: KimiStreamOptions {
            include_usage: true,
        },
        max_tokens: 32,
        reasoning_effort: None,
        thinking: None,
        prompt_cache_key: None,
    };
    assert_eq!(kimi.model, "kimi-for-coding");
}
