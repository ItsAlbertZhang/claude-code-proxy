---
title: Monitor TUI
description: Use the claude-code-proxy monitor to inspect sessions, active and recent requests, providers, errors, token usage, throughput, and setup.
---

`claude-code-proxy serve` opens the monitor when stdout is an interactive terminal. The same process runs the HTTP listener.

![claude-code-proxy monitor showing sessions, active requests, recent requests, and events](/monitor-tui.webp)

## What the monitor shows

- Sessions grouped by Claude Code session ID and project
- Active request lifecycle and selected provider or model
- Recent requests, HTTP status, elapsed time, and errors
- Input and output token totals
- Output throughput based on matched upstream timing and cumulative usage samples
- Paths to traffic captures when capture is enabled
- Configuration overrides and a ready-to-copy Claude Code setup

## Keyboard controls

| Key | Action |
| --- | --- |
| `Tab`, `←`, `→` | Change focused pane |
| `j`, `k`, `↓`, `↑` | Move selection |
| `Enter` | Open session or request details |
| `Esc` | Close details or an overlay |
| `?` | Toggle shortcut help |
| `b` | Toggle the setup overlay |
| `q` | Request a graceful shutdown |
| `Ctrl-C` | Force shutdown |

The request table changes columns as the terminal width changes.

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

## Background services

### Linux systemd user service

Direct Linux release archives and the release installer include `ccpd`, a small systemd user-service manager. Authenticate interactively before starting the service:

```sh
claude-code-proxy codex auth login
ccpd install --now
```

The install command writes `${XDG_CONFIG_HOME:-~/.config}/systemd/user/claude-code-proxy.service`, enables it for future logins, and starts it immediately. It records the absolute path of the current `claude-code-proxy` executable, so run `ccpd install --now` again after moving the binary.

Common management commands are:

```sh
ccpd status
ccpd health
ccpd logs
ccpd restart
ccpd uninstall
```

`ccpd logs` follows the user journal. To keep the user manager running after logout and start it during boot, enable lingering once:

```sh
loginctl enable-linger "$USER"
```

Optional service environment variables use systemd `KEY=value` syntax, without `export`, in `${XDG_CONFIG_HOME:-~/.config}/claude-code-proxy/ccpd.env`. Keep this file private:

```sh
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/claude-code-proxy"
chmod 700 "${XDG_CONFIG_HOME:-$HOME/.config}/claude-code-proxy"
$EDITOR "${XDG_CONFIG_HOME:-$HOME/.config}/claude-code-proxy/ccpd.env"
chmod 600 "${XDG_CONFIG_HOME:-$HOME/.config}/claude-code-proxy/ccpd.env"
ccpd restart
```

The service can write only to the standard claude-code-proxy configuration and state directories under its home-directory sandbox. Do not override `HOME`, `XDG_CONFIG_HOME`, `XDG_STATE_HOME`, or `CCP_CONFIG_DIR` in `ccpd.env`. The filesystem sandbox for user services requires unprivileged user namespaces; distributions that disable them cannot start this unit.

When installing from a source checkout rather than a Linux release, install the helper separately:

```sh
install -Dm755 scripts/ccpd "$HOME/.local/bin/ccpd"
```

### Homebrew

Homebrew manages the background process itself and does not install `ccpd`:

```sh
brew services start claude-code-proxy
```

Homebrew service output lives in `~/.local/state/claude-code-proxy/service.log` on macOS and Linux. The structured `proxy.log` shares the state directory. Provider login remains an interactive one-time command.
