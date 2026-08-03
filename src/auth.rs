use anyhow::Result;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::to_string_pretty;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::marker::PhantomData;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

pub trait AuthStorage<T>: Send + Sync
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone,
{
    fn load(&self) -> Result<Option<T>>;
    fn save(&self, value: T) -> Result<()>;
    fn clear(&self) -> Result<()>;
    fn compare_and_swap(&self, expected: Option<&T>, replacement: Option<T>) -> Result<bool> {
        let current = self.load()?;
        if !serialized_values_equal(current.as_ref(), expected)? {
            return Ok(false);
        }
        match replacement {
            Some(value) => self.save(value)?,
            None => self.clear()?,
        }
        Ok(true)
    }
    fn path(&self) -> String;
}

fn serialized_values_equal<T: Serialize>(left: Option<&T>, right: Option<&T>) -> Result<bool> {
    Ok(match (left, right) {
        (Some(left), Some(right)) => serde_json::to_value(left)? == serde_json::to_value(right)?,
        (None, None) => true,
        _ => false,
    })
}

pub trait Keychain: Send + Sync {
    fn read(&self, service: &str, account: &str) -> Result<Option<String>>;
    fn write(&self, service: &str, account: &str, value: &str) -> Result<()>;
    fn delete(&self, service: &str, account: &str) -> Result<()>;
}

#[derive(Default)]
pub struct StubKeychain;

impl Keychain for StubKeychain {
    fn read(&self, _service: &str, _account: &str) -> Result<Option<String>> {
        Ok(None)
    }

    fn write(&self, _service: &str, _account: &str, _value: &str) -> Result<()> {
        Ok(())
    }

    fn delete(&self, _service: &str, _account: &str) -> Result<()> {
        Ok(())
    }
}

#[derive(Default, Clone, Copy)]
pub struct SystemKeychain;

#[cfg(target_os = "macos")]
impl Keychain for SystemKeychain {
    fn read(&self, service: &str, account: &str) -> Result<Option<String>> {
        let output = run_security(&["find-generic-password", "-s", service, "-a", account, "-w"])?;
        if output.status.success() {
            let mut raw = String::from_utf8(output.stdout)
                .map_err(|err| anyhow::anyhow!("Keychain value is not valid UTF-8: {err}"))?;
            trim_one_trailing_newline(&mut raw);
            return Ok(Some(raw));
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("could not be found") || stderr.contains("specified item could not") {
            return Ok(None);
        }
        Err(anyhow::anyhow!("Keychain read failed: {}", stderr.trim()))
    }

    fn write(&self, _service: &str, _account: &str, _value: &str) -> Result<()> {
        anyhow::bail!("Keychain write is not available through non-interactive compatibility mode")
    }

    fn delete(&self, service: &str, account: &str) -> Result<()> {
        let output = run_security(&["delete-generic-password", "-s", service, "-a", account])?;
        if output.status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("could not be found") || stderr.contains("specified item could not") {
            return Ok(());
        }
        Err(anyhow::anyhow!("Keychain delete failed: {}", stderr.trim()))
    }
}

#[cfg(target_os = "macos")]
fn run_security(args: &[&str]) -> Result<std::process::Output> {
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    let mut child = Command::new("/usr/bin/security")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| anyhow::anyhow!("Failed to start /usr/bin/security: {err}"))?;
    let start = Instant::now();
    loop {
        if child
            .try_wait()
            .map_err(|err| anyhow::anyhow!("Failed waiting for /usr/bin/security: {err}"))?
            .is_some()
        {
            return child.wait_with_output().map_err(|err| {
                anyhow::anyhow!("Failed collecting /usr/bin/security output: {err}")
            });
        }
        if start.elapsed() >= Duration::from_secs(10) {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("Timed out reading macOS Keychain");
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

#[cfg(target_os = "macos")]
fn trim_one_trailing_newline(value: &mut String) {
    if value.ends_with('\n') {
        value.pop();
        if value.ends_with('\r') {
            value.pop();
        }
    }
}

#[cfg(not(target_os = "macos"))]
impl Keychain for SystemKeychain {
    fn read(&self, _service: &str, _account: &str) -> Result<Option<String>> {
        Ok(None)
    }

    fn write(&self, _service: &str, _account: &str, _value: &str) -> Result<()> {
        anyhow::bail!("Keychain storage is not available on this platform")
    }

    fn delete(&self, _service: &str, _account: &str) -> Result<()> {
        Ok(())
    }
}

fn with_auth_file_lock<R>(path: &str, action: impl FnOnce() -> Result<R>) -> Result<R> {
    let lock_path = format!("{path}.lock");
    let lock_path = std::path::Path::new(&lock_path);
    if let Some(directory) = lock_path.parent() {
        fs::create_dir_all(directory)?;
        set_mode(directory, 0o700);
    }
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)?;
    fs2::FileExt::lock_exclusive(&lock)?;
    let result = action();
    let unlock_result = fs2::FileExt::unlock(&lock);
    match (result, unlock_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error.into()),
    }
}

pub struct FileAuthStore<T>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone,
{
    file: String,
    legacy_file: String,
    _marker: std::marker::PhantomData<T>,
}

impl<T> FileAuthStore<T>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone,
{
    pub fn new(file: String, legacy_file: String) -> Self {
        Self {
            file,
            legacy_file,
            _marker: Default::default(),
        }
    }

    fn load_unlocked(&self) -> Option<T> {
        let parsed = load_auth_file::<T>(&self.file);
        if parsed.is_some() {
            return parsed;
        }
        if self.file == self.legacy_file {
            return None;
        }
        load_auth_file::<T>(&self.legacy_file)
    }

    fn save_unlocked(&self, value: &T) -> Result<()> {
        let path = std::path::Path::new(&self.file);
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
            set_mode(dir, 0o700);
        }
        write_atomically(&self.file, value)
    }

    fn clear_unlocked(&self) -> Result<()> {
        for path in [&self.file, &self.legacy_file] {
            if let Err(err) = fs::remove_file(path)
                && err.kind() != io::ErrorKind::NotFound
            {
                return Err(anyhow::Error::from(err));
            }
        }
        Ok(())
    }
}

impl<T> AuthStorage<T> for FileAuthStore<T>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone,
{
    fn load(&self) -> Result<Option<T>> {
        with_auth_file_lock(&self.file, || Ok(self.load_unlocked()))
    }

    fn save(&self, value: T) -> Result<()> {
        with_auth_file_lock(&self.file, || self.save_unlocked(&value))
    }

    fn clear(&self) -> Result<()> {
        with_auth_file_lock(&self.file, || self.clear_unlocked())
    }

    fn compare_and_swap(&self, expected: Option<&T>, replacement: Option<T>) -> Result<bool> {
        with_auth_file_lock(&self.file, || {
            let current = self.load_unlocked();
            if !serialized_values_equal(current.as_ref(), expected)? {
                return Ok(false);
            }
            match replacement.as_ref() {
                Some(value) => self.save_unlocked(value)?,
                None => self.clear_unlocked()?,
            }
            Ok(true)
        })
    }

    fn path(&self) -> String {
        self.file.clone()
    }
}

pub struct KeychainFileAuthStore<T, K = SystemKeychain>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone,
    K: Keychain,
{
    file_store: FileAuthStore<T>,
    keychain: K,
    service: String,
    account: String,
    use_keychain: bool,
    keychain_path: String,
    _marker: PhantomData<T>,
}

impl<T, K> KeychainFileAuthStore<T, K>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone,
    K: Keychain,
{
    pub fn new(
        file: String,
        legacy_file: String,
        service: impl Into<String>,
        account: impl Into<String>,
        use_keychain: bool,
        keychain: K,
    ) -> Self {
        Self {
            file_store: FileAuthStore::new(file, legacy_file),
            keychain,
            service: service.into(),
            account: account.into(),
            use_keychain,
            keychain_path: "macOS Keychain".to_string(),
            _marker: PhantomData,
        }
    }

    fn load_unlocked(&self) -> Result<Option<T>> {
        if let Some(parsed) = self.file_store.load_unlocked() {
            return Ok(Some(parsed));
        }
        if self.use_keychain
            && let Some(raw) = self.keychain.read(&self.service, &self.account)?
        {
            return serde_json::from_str::<T>(&raw)
                .map(Some)
                .map_err(|err| anyhow::anyhow!("Failed to parse Keychain auth JSON: {err}"));
        }
        Ok(None)
    }

    fn save_unlocked(&self, value: &T) -> Result<()> {
        if self.use_keychain {
            let raw = serde_json::to_string(value)?;
            if self
                .keychain
                .write(&self.service, &self.account, &raw)
                .is_ok()
            {
                return Ok(());
            }
        }
        self.file_store.save_unlocked(value)
    }

    fn clear_unlocked(&self) -> Result<()> {
        if self.use_keychain {
            self.keychain.delete(&self.service, &self.account)?;
        }
        self.file_store.clear_unlocked()
    }
}

impl<T, K> AuthStorage<T> for KeychainFileAuthStore<T, K>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone,
    K: Keychain,
{
    fn load(&self) -> Result<Option<T>> {
        with_auth_file_lock(&self.file_store.file, || self.load_unlocked())
    }

    fn save(&self, value: T) -> Result<()> {
        with_auth_file_lock(&self.file_store.file, || self.save_unlocked(&value))
    }

    fn clear(&self) -> Result<()> {
        with_auth_file_lock(&self.file_store.file, || self.clear_unlocked())
    }

    fn compare_and_swap(&self, expected: Option<&T>, replacement: Option<T>) -> Result<bool> {
        with_auth_file_lock(&self.file_store.file, || {
            let current = self.load_unlocked()?;
            if !serialized_values_equal(current.as_ref(), expected)? {
                return Ok(false);
            }
            match replacement.as_ref() {
                Some(value) => self.save_unlocked(value)?,
                None => self.clear_unlocked()?,
            }
            Ok(true)
        })
    }

    fn path(&self) -> String {
        if self.use_keychain {
            self.keychain_path.clone()
        } else {
            self.file_store.path()
        }
    }
}

pub fn load_auth_file<T: DeserializeOwned>(path: &str) -> Option<T> {
    let mut file = File::open(path).ok()?;
    let mut raw = String::new();
    file.read_to_string(&mut raw).ok()?;
    serde_json::from_str::<T>(&raw).ok()
}

pub fn load_auth_file_value(path: &std::path::Path) -> Option<serde_json::Value> {
    let mut file = File::open(path).ok()?;
    let mut raw = String::new();
    file.read_to_string(&mut raw).ok()?;
    serde_json::from_str::<serde_json::Value>(&raw).ok()
}

pub fn load_auth_file_with_legacy<T: DeserializeOwned>(
    primary: &std::path::Path,
    legacy: &std::path::Path,
) -> Option<T> {
    if let Some(value) = load_auth_file_value(primary) {
        return serde_json::from_value(value).ok();
    }
    if primary == legacy {
        None
    } else {
        load_auth_file_value(legacy).and_then(|value| serde_json::from_value(value).ok())
    }
}

pub fn delete_auth_file(primary: &std::path::Path, legacy: &std::path::Path) -> io::Result<()> {
    if let Err(err) = fs::remove_file(primary)
        && err.kind() != io::ErrorKind::NotFound
    {
        return Err(err);
    }
    if primary != legacy
        && let Err(err) = fs::remove_file(legacy)
        && err.kind() != io::ErrorKind::NotFound
    {
        return Err(err);
    }
    Ok(())
}

pub fn write_atomically<T: Serialize>(path: &str, value: &T) -> Result<()> {
    let dir = std::path::Path::new(path)
        .parent()
        .ok_or_else(|| anyhow::anyhow!("invalid auth path"))?;
    fs::create_dir_all(dir)?;
    set_mode(dir, 0o700);

    let tmp = format!("{path}.tmp-{}", uuid::Uuid::new_v4());
    #[cfg(unix)]
    let mut out = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)?
    };
    #[cfg(not(unix))]
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)?;
    out.write_all(to_string_pretty(value)?.as_bytes())?;
    out.sync_all()?;
    if let Err(err) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(err.into());
    }
    set_mode(std::path::Path::new(path), 0o600);
    Ok(())
}

fn set_mode(_path: &std::path::Path, _mode: u32) {
    #[cfg(unix)]
    {
        if let Ok(meta) = fs::metadata(_path) {
            let mut permissions = meta.permissions();
            permissions.set_mode(_mode);
            let _ = fs::set_permissions(_path, permissions);
        }
    }
}

pub struct InMemoryAuthStore<T>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone,
{
    inner: std::sync::Arc<std::sync::Mutex<Option<T>>>,
}

impl<T> Default for InMemoryAuthStore<T>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> InMemoryAuthStore<T>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone,
{
    pub fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }
}

impl<T> Clone for InMemoryAuthStore<T>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> AuthStorage<T> for InMemoryAuthStore<T>
where
    T: Serialize + DeserializeOwned + Send + Sync + Clone,
{
    fn load(&self) -> Result<Option<T>> {
        let inner = self
            .inner
            .lock()
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        Ok(inner.clone())
    }

    fn save(&self, value: T) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        *inner = Some(value);
        Ok(())
    }

    fn clear(&self) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        *inner = None;
        Ok(())
    }

    fn compare_and_swap(&self, expected: Option<&T>, replacement: Option<T>) -> Result<bool> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        if !serialized_values_equal(inner.as_ref(), expected)? {
            return Ok(false);
        }
        *inner = replacement;
        Ok(true)
    }

    fn path(&self) -> String {
        "memory".to_string()
    }
}

#[cfg(test)]
pub fn fixture_store<T>() -> InMemoryAuthStore<T>
where
    T: Serialize + serde::de::DeserializeOwned + Send + Sync + Clone,
{
    InMemoryAuthStore::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    #[test]
    fn file_store_compare_and_swap_rejects_stale_writer() {
        let path = std::env::temp_dir()
            .join(format!("ccp-auth-cas-{}.json", uuid::Uuid::new_v4()))
            .to_string_lossy()
            .into_owned();
        let first = FileAuthStore::<serde_json::Value>::new(path.clone(), path.clone());
        let second = FileAuthStore::<serde_json::Value>::new(path.clone(), path.clone());
        let original = json!({"account":"a","access":"old"});
        let switched = json!({"account":"b","access":"new"});
        first.save(original.clone()).unwrap();
        second.save(switched.clone()).unwrap();

        assert!(
            !first
                .compare_and_swap(
                    Some(&original),
                    Some(json!({"account":"a","access":"stale"})),
                )
                .unwrap()
        );
        assert_eq!(first.load().unwrap(), Some(switched));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path}.lock"));
    }

    #[derive(Clone, Default)]
    struct MockKeychain {
        values: Arc<Mutex<HashMap<(String, String), String>>>,
    }

    impl MockKeychain {
        fn set_raw(&self, service: &str, account: &str, value: serde_json::Value) {
            self.values.lock().unwrap().insert(
                (service.to_string(), account.to_string()),
                value.to_string(),
            );
        }

        fn raw(&self, service: &str, account: &str) -> Option<String> {
            self.values
                .lock()
                .unwrap()
                .get(&(service.to_string(), account.to_string()))
                .cloned()
        }
    }

    impl Keychain for MockKeychain {
        fn read(&self, service: &str, account: &str) -> Result<Option<String>> {
            Ok(self.raw(service, account))
        }

        fn write(&self, service: &str, account: &str, value: &str) -> Result<()> {
            self.values.lock().unwrap().insert(
                (service.to_string(), account.to_string()),
                value.to_string(),
            );
            Ok(())
        }

        fn delete(&self, service: &str, account: &str) -> Result<()> {
            self.values
                .lock()
                .unwrap()
                .remove(&(service.to_string(), account.to_string()));
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct ReadOnlyKeychain(MockKeychain);

    impl Keychain for ReadOnlyKeychain {
        fn read(&self, service: &str, account: &str) -> Result<Option<String>> {
            self.0.read(service, account)
        }

        fn write(&self, _service: &str, _account: &str, _value: &str) -> Result<()> {
            anyhow::bail!("read-only")
        }

        fn delete(&self, service: &str, account: &str) -> Result<()> {
            self.0.delete(service, account)
        }
    }

    fn temp_auth_path(dir: &tempfile::TempDir, name: &str) -> String {
        dir.path().join(name).to_string_lossy().to_string()
    }

    #[test]
    fn keychain_file_store_loads_file_before_keychain() {
        let temp = tempfile::TempDir::new().unwrap();
        let file = temp_auth_path(&temp, "auth.json");
        let legacy = temp_auth_path(&temp, "legacy.json");
        write_atomically(&file, &json!({"source": "file"})).unwrap();

        let keychain = MockKeychain::default();
        keychain.set_raw("svc", "acct", json!({"source": "keychain"}));

        let store: KeychainFileAuthStore<serde_json::Value, _> =
            KeychainFileAuthStore::new(file, legacy, "svc", "acct", true, keychain);

        let loaded = store.load().unwrap().unwrap();
        assert_eq!(loaded["source"], json!("file"));
        assert_eq!(store.path(), "macOS Keychain");
    }

    #[test]
    fn keychain_file_store_falls_back_to_keychain_when_file_missing() {
        let temp = tempfile::TempDir::new().unwrap();
        let file = temp_auth_path(&temp, "auth.json");
        let legacy = temp_auth_path(&temp, "legacy.json");
        let keychain = MockKeychain::default();
        keychain.set_raw("svc", "acct", json!({"source": "keychain"}));

        let store: KeychainFileAuthStore<serde_json::Value, _> =
            KeychainFileAuthStore::new(file, legacy, "svc", "acct", true, keychain);

        let loaded = store.load().unwrap().unwrap();
        assert_eq!(loaded["source"], json!("keychain"));
    }

    #[test]
    fn keychain_file_store_saves_and_clears_keychain_when_enabled() {
        let temp = tempfile::TempDir::new().unwrap();
        let file = temp_auth_path(&temp, "auth.json");
        let legacy = temp_auth_path(&temp, "legacy.json");
        write_atomically(&file, &json!({"source": "file"})).unwrap();

        let keychain = MockKeychain::default();
        let store: KeychainFileAuthStore<serde_json::Value, _> =
            KeychainFileAuthStore::new(file.clone(), legacy, "svc", "acct", true, keychain.clone());

        store.save(json!({"source": "saved"})).unwrap();
        let raw = keychain.raw("svc", "acct").unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&raw).unwrap()["source"],
            json!("saved")
        );

        store.clear().unwrap();
        assert!(keychain.raw("svc", "acct").is_none());
        assert!(!std::path::Path::new(&file).exists());
    }

    #[test]
    fn keychain_file_store_falls_back_to_file_when_keychain_write_fails() {
        let temp = tempfile::TempDir::new().unwrap();
        let file = temp_auth_path(&temp, "auth.json");
        let legacy = temp_auth_path(&temp, "legacy.json");
        let store: KeychainFileAuthStore<serde_json::Value, _> = KeychainFileAuthStore::new(
            file.clone(),
            legacy,
            "svc",
            "acct",
            true,
            ReadOnlyKeychain::default(),
        );

        store.save(json!({"source": "file-fallback"})).unwrap();
        assert_eq!(
            store.load().unwrap().unwrap()["source"],
            json!("file-fallback")
        );
        assert!(std::path::Path::new(&file).exists());
    }

    #[test]
    fn keychain_file_store_uses_file_when_keychain_disabled() {
        let temp = tempfile::TempDir::new().unwrap();
        let file = temp_auth_path(&temp, "auth.json");
        let legacy = temp_auth_path(&temp, "legacy.json");
        let keychain = MockKeychain::default();
        let store: KeychainFileAuthStore<serde_json::Value, _> = KeychainFileAuthStore::new(
            file.clone(),
            legacy,
            "svc",
            "acct",
            false,
            keychain.clone(),
        );

        store.save(json!({"source": "file"})).unwrap();
        assert!(keychain.raw("svc", "acct").is_none());
        assert_eq!(store.path(), file);
        assert_eq!(store.load().unwrap().unwrap()["source"], json!("file"));
    }
}
