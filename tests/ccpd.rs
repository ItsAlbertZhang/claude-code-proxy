#![cfg(target_os = "linux")]

use std::error::Error;
use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

const UNIT_NAME: &str = "claude-code-proxy.service";

struct Fixture {
    temp: TempDir,
    home: PathBuf,
    config_home: PathBuf,
    fake_bin: PathBuf,
    systemctl_log: PathBuf,
    journalctl_log: PathBuf,
    curl_log: PathBuf,
    proxy_log: PathBuf,
    proxy_v1: PathBuf,
    proxy_v2: PathBuf,
}

impl Fixture {
    fn new() -> Result<Self, Box<dyn Error>> {
        let temp = TempDir::new()?;
        let home = temp.path().join("home");
        let config_home = temp.path().join("xdg-config");
        let fake_bin = temp.path().join("fake-bin");
        fs::create_dir_all(&home)?;
        fs::create_dir_all(&config_home)?;
        fs::create_dir_all(&fake_bin)?;

        let systemctl_log = temp.path().join("systemctl.log");
        let journalctl_log = temp.path().join("journalctl.log");
        let curl_log = temp.path().join("curl.log");
        let proxy_log = temp.path().join("proxy.log");

        write_executable(
            &fake_bin.join("systemctl"),
            r#"#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >>"$FAKE_SYSTEMCTL_LOG"
case "$*" in
  "--user show-environment")
    if [[ -n "${FAKE_MANAGER_HOME_SERIALIZED+x}" ]]; then
      printf 'HOME=%s\n' "$FAKE_MANAGER_HOME_SERIALIZED"
    elif [[ -n "${FAKE_MANAGER_HOME+x}" ]]; then
      printf 'HOME=%s\n' "$FAKE_MANAGER_HOME"
    fi
    if [[ -n "${FAKE_MANAGER_XDG_CONFIG_HOME_SERIALIZED+x}" ]]; then
      printf 'XDG_CONFIG_HOME=%s\n' "$FAKE_MANAGER_XDG_CONFIG_HOME_SERIALIZED"
    elif [[ -n "${FAKE_MANAGER_XDG_CONFIG_HOME+x}" ]]; then
      printf 'XDG_CONFIG_HOME=%s\n' "$FAKE_MANAGER_XDG_CONFIG_HOME"
    fi
    exit "${FAKE_SYSTEMD_AVAILABLE_RC:-0}"
    ;;
  "--user is-enabled --quiet claude-code-proxy.service")
    exit "${FAKE_IS_ENABLED_RC:-1}"
    ;;
  "--user is-active --quiet claude-code-proxy.service")
    exit "${FAKE_IS_ACTIVE_RC:-1}"
    ;;
esac
exit "${FAKE_SYSTEMCTL_RC:-0}"
"#,
        )?;
        write_executable(
            &fake_bin.join("journalctl"),
            r#"#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >>"$FAKE_JOURNALCTL_LOG"
printf '%s\n' "${FAKE_JOURNALCTL_OUTPUT:-journal output}"
exit "${FAKE_JOURNALCTL_RC:-0}"
"#,
        )?;
        write_executable(
            &fake_bin.join("curl"),
            r#"#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >>"$FAKE_CURL_LOG"
printf '%s' "${FAKE_CURL_OUTPUT:-{\"ok\":true}}"
exit "${FAKE_CURL_RC:-0}"
"#,
        )?;

        let proxy_v1 = temp
            .path()
            .join("proxy version one")
            .join("claude-code-proxy");
        let proxy_v2 = temp
            .path()
            .join("proxy version two")
            .join("claude-code-proxy");
        write_fake_proxy(&proxy_v1, "claude-code-proxy test-v1")?;
        write_fake_proxy(&proxy_v2, "claude-code-proxy test-v2")?;

        Ok(Self {
            temp,
            home,
            config_home,
            fake_bin,
            systemctl_log,
            journalctl_log,
            curl_log,
            proxy_log,
            proxy_v1,
            proxy_v2,
        })
    }

    fn command(&self, args: &[&str]) -> Command {
        self.command_with_proxy(args, &self.proxy_v1)
    }

    fn command_with_proxy(&self, args: &[&str], proxy: &Path) -> Command {
        let mut command = Command::new(ccpd_script());
        command
            .args(args)
            .env_clear()
            .env("PATH", format!("{}:/usr/bin:/bin", self.fake_bin.display()))
            .env("HOME", &self.home)
            .env("XDG_CONFIG_HOME", &self.config_home)
            .env("CCPD_PROXY_BIN", proxy)
            .env("FAKE_SYSTEMCTL_LOG", &self.systemctl_log)
            .env("FAKE_JOURNALCTL_LOG", &self.journalctl_log)
            .env("FAKE_CURL_LOG", &self.curl_log)
            .env("FAKE_PROXY_LOG", &self.proxy_log)
            .env("FAKE_MANAGER_HOME", &self.home)
            .env("FAKE_MANAGER_XDG_CONFIG_HOME", &self.config_home);
        command
    }

    fn run(&self, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        Ok(self.command(args).output()?)
    }

    fn unit_path(&self) -> PathBuf {
        Self::unit_path_at(&self.config_home)
    }

    fn unit_path_at(config_home: &Path) -> PathBuf {
        config_home.join("systemd").join("user").join(UNIT_NAME)
    }

    fn clear_calls(&self) -> Result<(), Box<dyn Error>> {
        fs::write(&self.systemctl_log, "")?;
        fs::write(&self.journalctl_log, "")?;
        fs::write(&self.curl_log, "")?;
        fs::write(&self.proxy_log, "")?;
        Ok(())
    }
}

fn ccpd_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("ccpd")
}

fn write_executable(path: &Path, contents: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents)?;
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

fn write_fake_proxy(path: &Path, version: &str) -> Result<(), Box<dyn Error>> {
    write_executable(
        path,
        &format!(
            r#"#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >>"$FAKE_PROXY_LOG"
printf '%s\n' '{version}'
"#
        ),
    )
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn log_lines(path: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    match fs::read_to_string(path) {
        Ok(contents) => Ok(contents.lines().map(str::to_owned).collect()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err.into()),
    }
}

fn systemd_quote(value: &str, escape_dollars: bool) -> String {
    let mut escaped = if escape_dollars {
        value.replace('$', "$$")
    } else {
        value.to_owned()
    };
    escaped = escaped.replace('\\', "\\\\");
    escaped = escaped.replace('"', "\\\"");
    escaped = escaped.replace('%', "%%");
    format!("\"{escaped}\"")
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "status: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout(output),
        stderr(output)
    );
}

#[test]
fn install_now_renders_unit_and_enables_service() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new()?;
    let output = fixture.run(&["install", "--now"])?;
    assert_success(&output);

    let unit_path = fixture.unit_path();
    assert!(unit_path.is_file());
    assert_eq!(fs::metadata(&unit_path)?.mode() & 0o777, 0o644);

    let unit = fs::read_to_string(&unit_path)?;
    assert!(unit.contains("[Service]"), "{unit}");
    assert!(
        unit.contains(&format!(
            "ExecStart=\"{}\" serve --no-monitor",
            fixture.proxy_v1.display()
        )),
        "{unit}"
    );
    assert!(
        unit.contains("EnvironmentFile=-%E/claude-code-proxy/ccpd.env"),
        "{unit}"
    );
    assert!(unit.contains("WantedBy=default.target"), "{unit}");

    assert_eq!(
        log_lines(&fixture.systemctl_log)?,
        [
            "--user show-environment",
            "--user daemon-reload",
            "--user enable claude-code-proxy.service",
            "--user restart claude-code-proxy.service",
        ]
    );
    assert!(stdout(&output).contains(&format!("Installed {}", unit_path.display())));
    Ok(())
}

#[test]
fn reinstall_is_idempotent_but_still_reloads_systemd() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new()?;
    let first = fixture.run(&["install"])?;
    assert_success(&first);
    let before = fs::metadata(fixture.unit_path())?.ino();

    fixture.clear_calls()?;
    let second = fixture.run(&["install"])?;
    assert_success(&second);
    let after = fs::metadata(fixture.unit_path())?.ino();

    assert_eq!(before, after, "identical install replaced the unit file");
    assert_eq!(
        log_lines(&fixture.systemctl_log)?,
        ["--user show-environment", "--user daemon-reload"]
    );
    assert!(
        stdout(&second).contains("Run `ccpd enable`"),
        "{}",
        stdout(&second)
    );
    Ok(())
}

#[test]
fn reinstall_updates_changed_proxy_binary_path() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new()?;
    let first = fixture.run(&["install"])?;
    assert_success(&first);

    fixture.clear_calls()?;
    let second = fixture
        .command_with_proxy(&["install"], &fixture.proxy_v2)
        .output()?;
    assert_success(&second);

    let unit = fs::read_to_string(fixture.unit_path())?;
    assert!(
        !unit.contains(&fixture.proxy_v1.display().to_string()),
        "{unit}"
    );
    assert!(
        unit.contains(&format!(
            "ExecStart=\"{}\" serve --no-monitor",
            fixture.proxy_v2.display()
        )),
        "{unit}"
    );
    assert_eq!(
        log_lines(&fixture.systemctl_log)?,
        ["--user show-environment", "--user daemon-reload"]
    );
    Ok(())
}

#[test]
fn dollar_quoted_manager_home_and_xdg_override_shell_environment() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new()?;
    let manager_home = fixture.temp.path().join("manager home");
    let manager_config = fixture.temp.path().join("manager config");
    fs::create_dir_all(&manager_home)?;
    fs::create_dir_all(&manager_config)?;

    let mut command = fixture.command(&["install"]);
    command
        .env(
            "FAKE_MANAGER_HOME_SERIALIZED",
            format!("$'{}'", manager_home.display()),
        )
        .env(
            "FAKE_MANAGER_XDG_CONFIG_HOME_SERIALIZED",
            format!("$'{}'", manager_config.display()),
        );
    let output = command.output()?;
    assert_success(&output);

    let manager_unit_path = Fixture::unit_path_at(&manager_config);
    assert!(manager_unit_path.is_file());
    assert!(
        !fixture.unit_path().exists(),
        "shell XDG_CONFIG_HOME unexpectedly won"
    );

    let unit = fs::read_to_string(&manager_unit_path)?;
    let service_path = format!(
        "PATH={}:{}:{}:/home/linuxbrew/.linuxbrew/bin:/usr/local/bin:/usr/bin:/bin",
        fixture.proxy_v1.parent().unwrap().display(),
        manager_home.join(".local/bin").display(),
        manager_home.join(".cargo/bin").display()
    );
    assert!(
        unit.contains(&format!(
            "Environment={}",
            systemd_quote(&service_path, false)
        )),
        "{unit}"
    );
    assert!(
        !unit.contains(&fixture.home.display().to_string()),
        "{unit}"
    );
    assert_eq!(
        log_lines(&fixture.systemctl_log)?,
        ["--user show-environment", "--user daemon-reload"]
    );
    Ok(())
}

#[test]
fn dollar_quoted_manager_path_with_control_character_is_rejected() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new()?;
    let mut command = fixture.command(&["install"]);
    command.env(
        "FAKE_MANAGER_XDG_CONFIG_HOME_SERIALIZED",
        "$'/tmp/manager\\nconfig'",
    );
    let output = command.output()?;

    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("ccpd: XDG_CONFIG_HOME cannot contain control characters"),
        "{}",
        stderr(&output)
    );
    assert_eq!(
        log_lines(&fixture.systemctl_log)?,
        ["--user show-environment"]
    );
    Ok(())
}

#[test]
fn unit_escapes_special_executable_path_for_systemd() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new()?;
    let special_proxy = fixture
        .temp
        .path()
        .join(r#"proxy $dollar ${HOME} 50% "quote" back\slash"#)
        .join("claude-code-proxy");
    write_fake_proxy(&special_proxy, "claude-code-proxy special")?;

    let output = fixture
        .command_with_proxy(&["print-unit"], &special_proxy)
        .output()?;
    assert_success(&output);
    let unit = stdout(&output);

    let expected_exec = format!(
        "ExecStart={} serve --no-monitor",
        systemd_quote(&special_proxy.display().to_string(), true)
    );
    assert!(unit.contains(&expected_exec), "{unit}");

    let service_path = format!(
        "PATH={}:{}:{}:/home/linuxbrew/.linuxbrew/bin:/usr/local/bin:/usr/bin:/bin",
        special_proxy.parent().unwrap().display(),
        fixture.home.join(".local/bin").display(),
        fixture.home.join(".cargo/bin").display()
    );
    let expected_environment = format!("Environment={}", systemd_quote(&service_path, false));
    assert!(unit.contains(&expected_environment), "{unit}");

    assert!(expected_exec.contains("$$dollar"), "{expected_exec}");
    assert!(expected_exec.contains("$${HOME}"), "{expected_exec}");
    assert!(expected_exec.contains("50%%"), "{expected_exec}");
    assert!(expected_exec.contains(r#"\"quote\""#), "{expected_exec}");
    assert!(expected_exec.contains(r#"back\\slash"#), "{expected_exec}");
    Ok(())
}

#[test]
fn failed_daemon_reload_can_be_retried_after_unit_is_installed() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new()?;
    let mut first_command = fixture.command(&["install"]);
    first_command.env("FAKE_SYSTEMCTL_RC", "23");
    let first = first_command.output()?;
    assert_eq!(first.status.code(), Some(23));
    assert!(fixture.unit_path().is_file());
    assert_eq!(
        log_lines(&fixture.systemctl_log)?,
        ["--user show-environment", "--user daemon-reload"]
    );

    fixture.clear_calls()?;
    let retry = fixture.run(&["install"])?;
    assert_success(&retry);
    assert_eq!(
        log_lines(&fixture.systemctl_log)?,
        ["--user show-environment", "--user daemon-reload"]
    );
    Ok(())
}

#[test]
fn uninstall_removes_only_unit_and_preserves_environment_and_credentials()
-> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new()?;
    let install = fixture.run(&["install"])?;
    assert_success(&install);

    let proxy_config = fixture.config_home.join("claude-code-proxy");
    let env_path = proxy_config.join("ccpd.env");
    let credentials_path = proxy_config.join("codex").join("auth.json");
    fs::create_dir_all(credentials_path.parent().unwrap())?;
    fs::write(&env_path, "HTTPS_PROXY=http://127.0.0.1:7890\n")?;
    fs::write(&credentials_path, "{\"refresh\":\"keep-me\"}\n")?;

    fixture.clear_calls()?;
    let mut command = fixture.command(&["uninstall"]);
    command
        .env("FAKE_IS_ENABLED_RC", "1")
        .env("FAKE_IS_ACTIVE_RC", "0");
    let output = command.output()?;
    assert_success(&output);

    assert!(!fixture.unit_path().exists());
    assert_eq!(
        fs::read_to_string(&env_path)?,
        "HTTPS_PROXY=http://127.0.0.1:7890\n"
    );
    assert_eq!(
        fs::read_to_string(&credentials_path)?,
        "{\"refresh\":\"keep-me\"}\n"
    );
    assert_eq!(
        log_lines(&fixture.systemctl_log)?,
        [
            "--user show-environment",
            "--user is-enabled --quiet claude-code-proxy.service",
            "--user is-active --quiet claude-code-proxy.service",
            "--user disable --now claude-code-proxy.service",
            "--user daemon-reload",
        ]
    );
    assert!(
        stdout(&output).contains("credentials, configuration, and logs were left untouched"),
        "{}",
        stdout(&output)
    );

    fixture.clear_calls()?;
    let second = fixture.run(&["uninstall"])?;
    assert_success(&second);
    assert!(!fixture.unit_path().exists());
    assert_eq!(
        fs::read_to_string(&env_path)?,
        "HTTPS_PROXY=http://127.0.0.1:7890\n"
    );
    assert_eq!(
        fs::read_to_string(&credentials_path)?,
        "{\"refresh\":\"keep-me\"}\n"
    );
    assert_eq!(
        log_lines(&fixture.systemctl_log)?,
        [
            "--user show-environment",
            "--user is-enabled --quiet claude-code-proxy.service",
            "--user is-active --quiet claude-code-proxy.service",
            "--user daemon-reload",
        ]
    );
    Ok(())
}

#[test]
fn service_commands_forward_exact_systemd_arguments() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new()?;
    let cases: &[(&[&str], &[&str])] = &[
        (
            &["enable"],
            &[
                "--user show-environment",
                "--user daemon-reload",
                "--user enable --now claude-code-proxy.service",
            ],
        ),
        (
            &["disable"],
            &[
                "--user show-environment",
                "--user disable --now claude-code-proxy.service",
            ],
        ),
        (
            &["start"],
            &[
                "--user show-environment",
                "--user start claude-code-proxy.service",
            ],
        ),
        (
            &["stop"],
            &[
                "--user show-environment",
                "--user stop claude-code-proxy.service",
            ],
        ),
        (
            &["restart"],
            &[
                "--user show-environment",
                "--user restart claude-code-proxy.service",
            ],
        ),
        (
            &["status"],
            &[
                "--user show-environment",
                "--user status --no-pager --full claude-code-proxy.service",
            ],
        ),
    ];

    for (args, expected) in cases {
        fixture.clear_calls()?;
        let output = fixture.run(args)?;
        assert_success(&output);
        assert_eq!(log_lines(&fixture.systemctl_log)?, *expected, "{args:?}");
    }

    fixture.clear_calls()?;
    let output = fixture.run(&[])?;
    assert_success(&output);
    assert_eq!(
        log_lines(&fixture.systemctl_log)?,
        [
            "--user show-environment",
            "--user status --no-pager --full claude-code-proxy.service",
        ]
    );
    Ok(())
}

#[test]
fn logs_health_and_version_forward_to_their_tools() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new()?;

    let logs = fixture.run(&["logs", "--since", "today"])?;
    assert_success(&logs);
    assert_eq!(
        log_lines(&fixture.systemctl_log)?,
        ["--user show-environment"]
    );
    assert_eq!(
        log_lines(&fixture.journalctl_log)?,
        ["--user --unit=claude-code-proxy.service --follow --since today"]
    );

    fixture.clear_calls()?;
    let mut health_command = fixture.command(&["health"]);
    health_command.env("CCPD_HEALTH_URL", "http://127.0.0.1:19999/healthz");
    let health = health_command.output()?;
    assert_success(&health);
    assert_eq!(stdout(&health), "{\"ok\":true}\n");
    assert_eq!(
        log_lines(&fixture.curl_log)?,
        ["--fail --silent --show-error -- http://127.0.0.1:19999/healthz"]
    );

    fixture.clear_calls()?;
    let version = fixture.run(&["version"])?;
    assert_success(&version);
    assert_eq!(stdout(&version), "claude-code-proxy test-v1\n");
    assert_eq!(log_lines(&fixture.proxy_log)?, ["--version"]);
    Ok(())
}

#[test]
fn invalid_usage_and_unavailable_user_manager_report_errors() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new()?;

    let unknown = fixture.run(&["definitely-not-a-command"])?;
    assert_eq!(unknown.status.code(), Some(1));
    assert!(
        stderr(&unknown).contains("ccpd: unknown command: definitely-not-a-command"),
        "{}",
        stderr(&unknown)
    );

    let invalid_install = fixture.run(&["install", "--later"])?;
    assert_eq!(invalid_install.status.code(), Some(1));
    assert!(
        stderr(&invalid_install).contains("ccpd: install accepts only --now"),
        "{}",
        stderr(&invalid_install)
    );

    fixture.clear_calls()?;
    let mut unavailable = fixture.command(&["start"]);
    unavailable.env("FAKE_SYSTEMD_AVAILABLE_RC", "1");
    let unavailable = unavailable.output()?;
    assert_eq!(unavailable.status.code(), Some(1));
    assert!(
        stderr(&unavailable).contains("ccpd: the systemd user manager is unavailable"),
        "{}",
        stderr(&unavailable)
    );
    assert_eq!(
        log_lines(&fixture.systemctl_log)?,
        ["--user show-environment"]
    );

    let mut relative_proxy = fixture.command(&["print-unit"]);
    relative_proxy.env("CCPD_PROXY_BIN", "relative/claude-code-proxy");
    let relative_proxy = relative_proxy.output()?;
    assert_eq!(relative_proxy.status.code(), Some(1));
    assert!(
        stderr(&relative_proxy).contains("ccpd: proxy path must be an absolute path"),
        "{}",
        stderr(&relative_proxy)
    );

    let proxy_directory = fixture.temp.path().join("not-a-proxy-binary");
    fs::create_dir(&proxy_directory)?;
    let directory_proxy = fixture
        .command_with_proxy(&["print-unit"], &proxy_directory)
        .output()?;
    assert_eq!(directory_proxy.status.code(), Some(1));
    assert!(
        stderr(&directory_proxy).contains(&format!(
            "ccpd: proxy is not a regular file: {}",
            proxy_directory.display()
        )),
        "{}",
        stderr(&directory_proxy)
    );
    Ok(())
}
