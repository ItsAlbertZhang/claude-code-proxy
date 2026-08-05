---
title: Monitor TUI
description: Use the claude-code-proxy monitor to inspect sessions, agents, active and recent requests, models, errors, token usage, throughput, and setup.
---

`claude-code-proxy serve` opens the monitor when stdout is an interactive terminal. The same process runs the HTTP listener.

## What the monitor shows

- Sessions grouped by Claude Code session, with project context
- Active request lifecycle and the mapped model used upstream
- Recent requests, HTTP status, elapsed time, and errors
- Input and output token totals; active `Out` uses labelled stream bytes until positive upstream token usage is available
- Output throughput based on matched upstream timing and cumulative usage samples
- Paths to traffic captures when capture is enabled
- Configuration overrides and a ready-to-copy Claude Code setup

The Sessions table identifies each aggregate with the `ID` column. The `Agent` columns in request tables show a shortened Claude agent ID, `main` for a validated main-session request, and `-` for a stateless request. Provider routing stays internal. The `Model` columns show the mapped upstream model when available and fall back to the requested model until mapping completes. A Codex model whose final service tier is `priority` is prefixed with `f|`. The session and request detail views retain the requested model and show a distinct mapped model separately.

## Keyboard and mouse controls

| Key or input | Action |
| --- | --- |
| `Tab`, `←`, `→` | Change focused pane |
| `j`, `k`, `↓`, `↑` | Move selection, or scroll an open detail view |
| Mouse wheel | Move selection or scroll an open detail view one row per event |
| `Enter` | Open session or request details in the main content area |
| `Esc` | Close details or an overlay |
| `?` | Toggle shortcut help |
| `b` | Toggle the setup overlay |
| `q` | Request a graceful shutdown |
| `Ctrl-C` | Force shutdown |

The request table changes columns as the terminal width changes. Detail views replace the main table area; use the navigation keys or mouse wheel to scroll long diagnostics, and press `Esc` to return to the tables.

## Plain logs

Use plain output when the process runs under a service manager, in CI, or through a pipe:

```sh
claude-code-proxy serve --no-monitor
```

Non-terminal stdout also selects plain mode. `CCP_LOG_STDERR=1` mirrors JSONL log events to stderr in plain mode.

## Demo mode

Explore the full interface without binding a port or using provider credentials:

```sh
claude-code-proxy demo
```

The deterministic simulation covers active, successful, and failed requests across providers, projects, throughput states, and responsive layouts.

## Background service

A Homebrew installation can run at login:

```sh
brew services start claude-code-proxy
```

Service output lives in `~/.local/state/claude-code-proxy/service.log` on macOS and Linux. The structured `proxy.log` shares the state directory. Provider login remains an interactive one-time command.
