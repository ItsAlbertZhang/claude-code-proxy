use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::request_identity::ConversationIdentity;

use super::translate::request::ResponsesInputItem;

const SCHEMA_VERSION: u32 = 1;
const MAX_CHECKPOINT_BYTES: u64 = 4 * 1024 * 1024;
const AGENT_FILENAME_DOMAIN: &[u8] = b"claude-code-proxy/codex-compaction-agent/v1";

#[derive(Debug, thiserror::Error)]
pub(crate) enum CheckpointError {
    #[error("checkpoint I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("checkpoint JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
    #[error("checkpoint exceeds the size limit")]
    TooLarge,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum StoredOwner {
    Main {
        session_id: String,
    },
    Agent {
        session_id: String,
        agent_id: String,
    },
}

impl StoredOwner {
    fn from_identity(identity: &ConversationIdentity) -> Self {
        match identity {
            ConversationIdentity::Main(session_id) => Self::Main {
                session_id: session_id.clone(),
            },
            ConversationIdentity::Agent(session_id, agent_id) => Self::Agent {
                session_id: session_id.clone(),
                agent_id: agent_id.clone(),
            },
        }
    }

    fn session_id(&self) -> &str {
        match self {
            Self::Main { session_id } | Self::Agent { session_id, .. } => session_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CompactionCheckpoint {
    schema_version: u32,
    owner: StoredOwner,
    pub(crate) model: String,
    pub(crate) native_history: Vec<ResponsesInputItem>,
    pub(crate) portable_summary: String,
    updated_at_ms: u64,
}

impl CompactionCheckpoint {
    pub(crate) fn new(
        identity: &ConversationIdentity,
        model: String,
        native_history: Vec<ResponsesInputItem>,
        portable_summary: String,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            owner: StoredOwner::from_identity(identity),
            model,
            native_history,
            portable_summary,
            updated_at_ms: now_ms(),
        }
    }

    fn belongs_to(&self, identity: &ConversationIdentity) -> bool {
        self.schema_version == SCHEMA_VERSION
            && self.owner == StoredOwner::from_identity(identity)
            && self.owner.session_id() == identity.session_component()
    }
}

pub(crate) struct CheckpointStore {
    claude_dir: PathBuf,
    project_markers: Mutex<HashMap<String, PathBuf>>,
}

impl CheckpointStore {
    pub(crate) fn new(claude_dir: PathBuf) -> Self {
        Self {
            claude_dir,
            project_markers: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn save(
        &self,
        identity: &ConversationIdentity,
        checkpoint: &CompactionCheckpoint,
    ) -> Result<bool, CheckpointError> {
        if !checkpoint.belongs_to(identity) {
            return Ok(false);
        }
        let Some(path) = self.checkpoint_path(identity)? else {
            return Ok(false);
        };
        let mut encoded = serde_json::to_vec(checkpoint)?;
        encoded.push(b'\n');
        if encoded.len() as u64 > MAX_CHECKPOINT_BYTES {
            return Err(CheckpointError::TooLarge);
        }

        let directory = path
            .parent()
            .expect("checkpoint path always has a parent directory");
        let session_directory = directory
            .parent()
            .expect("checkpoint directory always belongs to a session directory");
        ensure_real_directory(session_directory)?;
        ensure_real_directory(directory)?;
        set_private_directory_permissions(directory)?;
        replace_file(&path, &encoded)?;
        Ok(true)
    }

    pub(crate) fn load(
        &self,
        identity: &ConversationIdentity,
    ) -> Result<Option<CompactionCheckpoint>, CheckpointError> {
        let Some(path) = self.checkpoint_path(identity)? else {
            return Ok(None);
        };
        if !checkpoint_parent_directories_are_real(&path)? {
            return Ok(None);
        }
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Ok(None);
        }
        if metadata.len() > MAX_CHECKPOINT_BYTES {
            return Err(CheckpointError::TooLarge);
        }
        let mut encoded = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_CHECKPOINT_BYTES + 1)
            .read_to_end(&mut encoded)?;
        if encoded.len() as u64 > MAX_CHECKPOINT_BYTES {
            return Err(CheckpointError::TooLarge);
        }
        let checkpoint: CompactionCheckpoint = serde_json::from_slice(&encoded)?;
        Ok(checkpoint.belongs_to(identity).then_some(checkpoint))
    }

    pub(crate) fn delete(&self, identity: &ConversationIdentity) -> Result<bool, CheckpointError> {
        let Some(path) = self.checkpoint_path(identity)? else {
            return Ok(false);
        };
        if !checkpoint_parent_directories_are_real(&path)? {
            return Ok(false);
        }
        match fs::remove_file(&path) {
            Ok(()) => {
                if let Some(directory) = path.parent() {
                    let _ = fs::remove_dir(directory);
                }
                Ok(true)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    fn checkpoint_path(
        &self,
        identity: &ConversationIdentity,
    ) -> Result<Option<PathBuf>, CheckpointError> {
        let session_id = identity.session_component();
        if !safe_session_directory_name(session_id) {
            return Ok(None);
        }
        let Some(marker) = self.resolve_session_marker(session_id)? else {
            return Ok(None);
        };
        let project_directory = marker
            .parent()
            .expect("resolved session marker always has a project directory");
        if !is_real_directory(project_directory)? {
            return Ok(None);
        }
        let session_dir = project_directory.join(session_id);
        Ok(Some(
            session_dir
                .join("codex-compaction")
                .join(checkpoint_filename(identity)),
        ))
    }

    fn resolve_session_marker(&self, session_id: &str) -> Result<Option<PathBuf>, CheckpointError> {
        let expected_name = OsString::from(format!("{session_id}.jsonl"));
        if let Some(marker) = self
            .project_markers
            .lock()
            .unwrap()
            .get(session_id)
            .cloned()
            && exact_regular_file(&marker, &expected_name)
        {
            if self.has_competing_session_marker(&expected_name, &marker)? {
                return Ok(None);
            }
            return Ok(Some(marker));
        }

        let projects = self.claude_dir.join("projects");
        let project_entries = match fs::read_dir(projects) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let mut matches = Vec::new();
        for project in project_entries {
            let project = project?;
            if !project.file_type()?.is_dir() {
                continue;
            }
            for entry in fs::read_dir(project.path())? {
                let entry = entry?;
                if entry.file_name() == expected_name && entry.file_type()?.is_file() {
                    matches.push(entry.path());
                    if matches.len() > 1 {
                        return Ok(None);
                    }
                }
            }
        }
        let Some(marker) = matches.pop() else {
            return Ok(None);
        };
        self.project_markers
            .lock()
            .unwrap()
            .insert(session_id.to_string(), marker.clone());
        Ok(Some(marker))
    }

    fn has_competing_session_marker(
        &self,
        expected_name: &OsString,
        cached_marker: &Path,
    ) -> Result<bool, CheckpointError> {
        let project_entries = match fs::read_dir(self.claude_dir.join("projects")) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        for project in project_entries {
            let project = project?;
            if !project.file_type()?.is_dir() {
                continue;
            }
            let candidate = project.path().join(expected_name);
            if candidate != cached_marker && exact_regular_file(&candidate, expected_name) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    #[cfg(test)]
    fn path_for_test(&self, identity: &ConversationIdentity) -> Option<PathBuf> {
        self.checkpoint_path(identity).unwrap()
    }
}

static CHECKPOINT_STORE: Lazy<CheckpointStore> =
    Lazy::new(|| CheckpointStore::new(crate::paths::claude_config_dir()));

pub(crate) fn save(
    identity: &ConversationIdentity,
    checkpoint: &CompactionCheckpoint,
) -> Result<bool, CheckpointError> {
    CHECKPOINT_STORE.save(identity, checkpoint)
}

pub(crate) fn load(
    identity: &ConversationIdentity,
) -> Result<Option<CompactionCheckpoint>, CheckpointError> {
    CHECKPOINT_STORE.load(identity)
}

pub(crate) fn delete(identity: &ConversationIdentity) -> Result<bool, CheckpointError> {
    CHECKPOINT_STORE.delete(identity)
}

fn checkpoint_filename(identity: &ConversationIdentity) -> String {
    match identity {
        ConversationIdentity::Main(_) => "main.json".to_string(),
        ConversationIdentity::Agent(_, agent_id) => {
            let mut digest = Sha256::new();
            digest.update(AGENT_FILENAME_DOMAIN);
            digest.update((agent_id.len() as u64).to_be_bytes());
            digest.update(agent_id.as_bytes());
            let encoded =
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest.finalize());
            format!("agent-{encoded}.json")
        }
    }
}

fn exact_regular_file(path: &Path, expected_name: &OsString) -> bool {
    path.file_name() == Some(expected_name.as_os_str())
        && fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_file())
}

fn is_real_directory(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(metadata.file_type().is_dir()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

fn checkpoint_parent_directories_are_real(path: &Path) -> io::Result<bool> {
    let directory = path
        .parent()
        .expect("checkpoint path always has a parent directory");
    let session_directory = directory
        .parent()
        .expect("checkpoint directory always belongs to a session directory");
    Ok(is_real_directory(session_directory)? && is_real_directory(directory)?)
}

fn safe_session_directory_name(session_id: &str) -> bool {
    !matches!(session_id, "." | "..")
        && !session_id.contains(['/', '\\'])
        && Path::new(session_id)
            .file_name()
            .is_some_and(|name| name == session_id)
}

#[cfg(not(windows))]
fn publish_temporary_file(temporary: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(temporary, destination)
}

#[cfg(windows)]
fn publish_temporary_file(temporary: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND};
    use windows_sys::Win32::Storage::FileSystem::{REPLACEFILE_WRITE_THROUGH, ReplaceFileW};

    let destination_wide = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let temporary_wide = temporary
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let replace = || unsafe {
        ReplaceFileW(
            destination_wide.as_ptr(),
            temporary_wide.as_ptr(),
            std::ptr::null(),
            REPLACEFILE_WRITE_THROUGH,
            std::ptr::null(),
            std::ptr::null(),
        )
    };

    if replace() != 0 {
        return Ok(());
    }
    let replace_error = io::Error::last_os_error();
    if !matches!(
        replace_error.raw_os_error(),
        Some(code)
            if code == ERROR_FILE_NOT_FOUND as i32 || code == ERROR_PATH_NOT_FOUND as i32
    ) {
        return Err(replace_error);
    }

    match fs::rename(temporary, destination) {
        Ok(()) => Ok(()),
        Err(_rename_error) if destination.exists() => {
            if replace() == 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        }
        Err(rename_error) => Err(rename_error),
    }
}

fn ensure_real_directory(path: &Path) -> io::Result<()> {
    match fs::create_dir(path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    if fs::symlink_metadata(path)?.file_type().is_dir() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint path component is not a real directory",
        ))
    }
}

fn replace_file(path: &Path, contents: &[u8]) -> io::Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");
    let temporary = path.with_file_name(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let result = (|| {
        let mut file = open_private_file(&temporary)?;
        file.write_all(contents)?;
        file.sync_all()?;
        drop(file);

        publish_temporary_file(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn open_private_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

fn set_private_directory_permissions(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn marker(root: &Path, project: &str, session: &str) {
        let project = root.join("projects").join(project);
        fs::create_dir_all(&project).unwrap();
        fs::write(project.join(format!("{session}.jsonl")), b"transcript\n").unwrap();
    }

    fn checkpoint(
        identity: &ConversationIdentity,
        encrypted_content: &str,
    ) -> CompactionCheckpoint {
        CompactionCheckpoint::new(
            identity,
            "gpt-5.6-sol".to_string(),
            vec![ResponsesInputItem::Compaction {
                encrypted_content: encrypted_content.to_string(),
            }],
            "portable summary with enough detail to identify this compacted conversation"
                .to_string(),
        )
    }

    #[test]
    fn main_and_agents_round_trip_exact_json_in_separate_files() {
        let root = tempfile::tempdir().unwrap();
        marker(root.path(), "project-a", "session-a");
        let store = CheckpointStore::new(root.path().to_path_buf());
        let main = ConversationIdentity::Main("session-a".to_string());
        let agent_a = ConversationIdentity::Agent("session-a".to_string(), "agent/a".to_string());
        let agent_b = ConversationIdentity::Agent("session-a".to_string(), "agent-b".to_string());
        let opaque = "opaque+/=\\\"\nsecond-line";

        for identity in [&main, &agent_a, &agent_b] {
            assert!(store.save(identity, &checkpoint(identity, opaque)).unwrap());
        }

        let main_path = store.path_for_test(&main).unwrap();
        let agent_a_path = store.path_for_test(&agent_a).unwrap();
        let agent_b_path = store.path_for_test(&agent_b).unwrap();
        assert_eq!(main_path.file_name().unwrap(), "main.json");
        assert_ne!(agent_a_path, agent_b_path);
        assert!(!agent_a_path.to_string_lossy().contains("agent/a"));
        assert_eq!(
            main_path.parent().unwrap().file_name().unwrap(),
            "codex-compaction"
        );

        for identity in [&main, &agent_a, &agent_b] {
            let loaded = store.load(identity).unwrap().unwrap();
            assert_eq!(loaded.model, "gpt-5.6-sol");
            assert_eq!(
                serde_json::to_value(loaded.native_history).unwrap(),
                json!([{"type":"compaction","encrypted_content":opaque}])
            );
        }
    }

    #[test]
    fn replacement_atomically_publishes_the_new_checkpoint() {
        let root = tempfile::tempdir().unwrap();
        marker(root.path(), "project-a", "session-a");
        let store = CheckpointStore::new(root.path().to_path_buf());
        let identity = ConversationIdentity::Main("session-a".to_string());

        assert!(
            store
                .save(&identity, &checkpoint(&identity, "first"))
                .unwrap()
        );
        assert!(
            store
                .save(&identity, &checkpoint(&identity, "second"))
                .unwrap()
        );

        let loaded = store.load(&identity).unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(loaded.native_history).unwrap(),
            json!([{"type":"compaction","encrypted_content":"second"}])
        );
    }

    #[test]
    fn unresolved_or_ambiguous_session_never_creates_checkpoint() {
        let root = tempfile::tempdir().unwrap();
        let store = CheckpointStore::new(root.path().to_path_buf());
        let identity = ConversationIdentity::Main("session-a".to_string());
        assert!(
            !store
                .save(&identity, &checkpoint(&identity, "opaque"))
                .unwrap()
        );

        marker(root.path(), "project-a", "session-a");
        marker(root.path(), "project-b", "session-a");
        assert!(
            !store
                .save(&identity, &checkpoint(&identity, "opaque"))
                .unwrap()
        );
        assert!(!root.path().join("projects/project-a/session-a").exists());
        assert!(!root.path().join("projects/project-b/session-a").exists());
    }

    #[test]
    fn cached_marker_fails_closed_if_session_becomes_ambiguous() {
        let root = tempfile::tempdir().unwrap();
        marker(root.path(), "project-a", "session-a");
        let store = CheckpointStore::new(root.path().to_path_buf());
        let identity = ConversationIdentity::Main("session-a".to_string());
        assert!(
            store
                .save(&identity, &checkpoint(&identity, "first"))
                .unwrap()
        );

        marker(root.path(), "project-b", "session-a");
        assert!(
            !store
                .save(&identity, &checkpoint(&identity, "second"))
                .unwrap()
        );
        assert!(store.load(&identity).unwrap().is_none());
    }

    #[test]
    fn dot_component_session_never_escapes_the_project_directory() {
        let root = tempfile::tempdir().unwrap();
        marker(root.path(), "project-a", "..");
        let store = CheckpointStore::new(root.path().to_path_buf());
        let identity = ConversationIdentity::Main("..".to_string());

        assert!(
            !store
                .save(&identity, &checkpoint(&identity, "opaque"))
                .unwrap()
        );
        assert!(!root.path().join("projects/codex-compaction").exists());
    }

    #[test]
    fn checkpoint_directory_does_not_follow_non_directory_session_component() {
        let root = tempfile::tempdir().unwrap();
        marker(root.path(), "project-a", "session-a");
        fs::write(
            root.path().join("projects/project-a/session-a"),
            b"not a directory",
        )
        .unwrap();
        let store = CheckpointStore::new(root.path().to_path_buf());
        let identity = ConversationIdentity::Main("session-a".to_string());

        assert!(matches!(
            store.save(&identity, &checkpoint(&identity, "opaque")),
            Err(CheckpointError::Io(error)) if error.kind() == io::ErrorKind::InvalidData
        ));
    }

    #[test]
    fn wrong_owner_and_corrupt_json_fail_closed_without_cross_loading() {
        let root = tempfile::tempdir().unwrap();
        marker(root.path(), "project-a", "session-a");
        let store = CheckpointStore::new(root.path().to_path_buf());
        let main = ConversationIdentity::Main("session-a".to_string());
        let agent = ConversationIdentity::Agent("session-a".to_string(), "agent-a".to_string());

        assert!(store.save(&main, &checkpoint(&main, "opaque")).unwrap());
        assert!(store.load(&agent).unwrap().is_none());
        fs::write(store.path_for_test(&main).unwrap(), b"{not-json").unwrap();
        assert!(matches!(store.load(&main), Err(CheckpointError::Json(_))));
    }

    #[test]
    fn oversized_checkpoint_is_rejected_before_json_parsing() {
        let root = tempfile::tempdir().unwrap();
        marker(root.path(), "project-a", "session-a");
        let store = CheckpointStore::new(root.path().to_path_buf());
        let identity = ConversationIdentity::Main("session-a".to_string());
        store
            .save(&identity, &checkpoint(&identity, "opaque"))
            .unwrap();
        fs::write(
            store.path_for_test(&identity).unwrap(),
            vec![b' '; MAX_CHECKPOINT_BYTES as usize + 1],
        )
        .unwrap();

        assert!(matches!(
            store.load(&identity),
            Err(CheckpointError::TooLarge)
        ));
    }

    #[test]
    fn delete_removes_only_the_selected_owner_checkpoint() {
        let root = tempfile::tempdir().unwrap();
        marker(root.path(), "project-a", "session-a");
        let store = CheckpointStore::new(root.path().to_path_buf());
        let main = ConversationIdentity::Main("session-a".to_string());
        let agent = ConversationIdentity::Agent("session-a".to_string(), "agent-a".to_string());
        store.save(&main, &checkpoint(&main, "main")).unwrap();
        store.save(&agent, &checkpoint(&agent, "agent")).unwrap();

        assert!(store.delete(&agent).unwrap());
        assert!(store.load(&agent).unwrap().is_none());
        assert!(store.load(&main).unwrap().is_some());
    }
}
