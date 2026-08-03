---
title: Compatibility and limitations
description: Canonical security, account, protocol, model, context, multimodal, reasoning, tool, session, rate-limit, and deployment boundaries.
---

claude-code-proxy targets Claude Code's practical Anthropic API usage rather than complete protocol equivalence.

## Accounts and provider policy

- Provider subscriptions, allowed models, regions, quotas, and enforcement remain under each provider's control.
- OpenAI has publicly welcomed using Codex through other coding harnesses, but public statements do not guarantee future policy or account treatment.
- Kimi, Grok, and Cursor use unofficial client integrations. Review the terms and account risk for your use.
- Upstream rate limits are shared with other clients on the same account.

## Listener security

- Incoming clients are not authenticated.
- The default bind address is `127.0.0.1`.
- A non-loopback listener requires an external firewall or authenticating reverse proxy.
- The proxy stores subscription credentials and can consume account quota, so an exposed listener is equivalent to exposing that capability.

## Anthropic API scope

- Messages supports streaming and non-streaming responses for the fields exercised by Claude Code.
- `?beta=true` does not select a separate implementation.
- Token counts are local estimates, not exact upstream tokenizer or billing counts.
- Claude Code title generation and other structured background requests are forwarded and consume provider tokens.
- Anthropic-specific fields without a provider mapping can be dropped.
- Native OpenAI Responses passthrough is opt-in and applies to registered Codex models.
- Codex Images passthrough is separately opt-in, restricted to `gpt-image-2`, and supports generation plus JSON or multipart edits. Variations, masks, remote image URLs, and URL-formatted outputs are unsupported.

## OpenAI API scope

- `CCP_CODEX_RESPONSES_API=1` enables `/v1/chat/completions` and `/v1/responses` for Codex, Kimi, Grok, and Cursor models.
- Codex Responses requests use native passthrough. Caller-supplied `previous_response_id` is preserved but prevents retrying a header 401 because the proxy does not own that native chain. A JSON or SSE unauthorized error encoded inside HTTP 200 preserves its status, bytes, and framing and can refresh credentials for only the next request. Requests for the other providers support text, reasoning, function tools, tool results, token limits, usage, streaming, aliases, and `[1m]` model hints.
- Unsupported non-null request fields return an error instead of being ignored.
- Grok search calls appear as Responses `web_search_call` items. Chat Completions returns the citations without a separate search item.
- Cursor tool bridging supports `Read`, `Write`, and `Bash`. It requires streaming and a valid stable conversational lane; sibling Agent lanes do not share pending tools.
- Stored Responses operations, WebSocket ingress, multiple choices, audio, log probabilities, and arbitrary hosted tools are unsupported.

## Models and context

- Local registration does not guarantee account access to a model.
- Unknown model IDs have no implicit provider fallback.
- Anthropic-style aliases route only to the configured Codex or Kimi alias provider.
- `[1m]` is a Claude Code client hint and does not change upstream context.
- Provider context limits can be lower than Claude Code's local threshold.
- Switching provider or model can clear provider-specific continuation assumptions while Claude Code retains portable history.

## Codex

- Base64 user images and supported base64 tool-result images map to Responses images. Remote URLs and malformed or unsupported nested images remain text placeholders.
- Reasoning summaries can appear as thinking blocks. Codex decides whether a summary is emitted. Encrypted reasoning and compaction items remain provider continuation data and are not exposed as raw chain of thought.
- Hosted web search supports mapped domain filters, but the Anthropic `max_uses` value is not enforced because Codex exposes no equivalent limit. Strict JSON schema output is translated; other Anthropic-only output settings can be omitted.
- A valid Claude session header selects Main. A valid direct Agent header selects an isolated child within that session; the parent header is lineage only. Missing, malformed, duplicate, or ambiguous identity tuples are stateless instead of falling back to Main.
- Provider state and upstream identifiers use domain-separated opaque lanes. Raw Claude session and Agent strings, OpenAI fallback IDs, and public Rust string-adapter values are compatibility input, not built-in provider keys or upstream conversation IDs.
- Translated GPT-5.6 Messages use Lite by default. The opt-in Full lane changes only Sol and Terra Messages plus their local count-token shape; Luna stays Lite. Native Responses and Chat ignore the flag and use their fixed model-based lanes. Hosted and standalone search use Full. Full and Lite state never mix.
- Append-only translated continuation requires the exact originating live WebSocket. A dead or replaced origin can fall back once to the complete translated input without a stale response ID; automatic transport fallback occurs only before an upstream request is sent.
- Each request binds an immutable endpoint, known account, credential, and protocol lane. Before semantic output, route A can rebuild exactly once as route B only for the same known account with changed credentials and replayable context. There is no route C. Unknown or changed account, unchanged credentials, a second 401, or any post-semantic failure is not replayed.
- Native caller-supplied `previous_response_id` and HTTP-200 JSON/SSE in-band unauthorized failures are never replayed. Anthropic live post-semantic authentication failures preserve published output and end with a sanitized Anthropic error.
- Count-token, detected auto-review classifier, image, transcription, and other auxiliary requests do not consume or mutate conversational affinity, sequence, continuation, WebSocket, compaction, Kimi, Cursor, or `Read` state.
- Native compaction is bound to the exact Main/direct-Agent owner, model, account/credential route, and Responses lane. It falls back to Claude Code's portable summary after mismatch, failure, eviction, restart, or 30 idle minutes. The registry caps at 1,000 entries, 4 MiB each, and 20,000,000 bytes total.
- Continuation and WebSocket registries are memory-only, expire after 30 idle minutes, and cap at 10,000 entries each. Continuation transcripts cap at 2 MB per owner and 20 MB total. `Read` correction metadata caps at 4,096 notes, survives a credential-route rollover on the same stable owner lane, and is isolated from sibling Agents across buffered and live output.

## Kimi

- The proxy exposes one Kimi Code wire model plus local aliases.
- Reasoning effort supports Kimi's low, medium, and high levels.
- Images in tool results use Kimi tool-message image parts.
- The persistent device ID is part of the login identity and must remain stable.

## Grok

- Model availability varies by account and region.
- Hosted general web search and X search are translated with citations and usage.
- The implemented multimodal path does not claim general image or video compatibility.

## Cursor Agent

- An installed Cursor Agent JavaScript bundle supplies protobuf classes.
- The dynamic catalog reflects the installed Cursor Agent and can differ across machines.
- The native tool bridge covers recognized `Read`, `Write`, and `Bash` calls when matching tools and a session ID are present.
- Cursor workspace callbacks and arbitrary native tool forms do not have a general Claude tool bridge.
- Conversation and pending tool state are in memory. Restarts clear them.
- Cursor count-tokens uses a rough rendered-prompt estimate.

## Diagnostics and privacy

- Structured logs redact known credential keys, but user-provided strings can still contain secrets.
- Error captures contain complete redacted failed responses.
- Traffic captures intentionally preserve prompts and tool content for message/Responses diagnostics.
- Image generation and edit routes never create traffic captures or persist prompts, uploads, data URLs, generated base64, or upstream error bodies.
- Verbose logging and traffic capture should be scoped to a focused local investigation.

For provider-specific behavior, use the [provider pages](/providers/choosing-a-provider/). For released changes, see the [Changelog](/reference/changelog/).
