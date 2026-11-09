use etcetera::{choose_app_strategy, AppStrategy, AppStrategyArgs};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

pub struct Paths;

impl Paths {
    fn get_dir(dir_type: DirType) -> PathBuf {
        if let Some(base) = Self::path_root() {
            match dir_type {
                DirType::Config => base.join("config"),
                DirType::Data => base.join("data"),
                DirType::State => base.join("state"),
                DirType::Plugins => base.join(".agents").join("plugins"),
                DirType::Agents => base.join(".agents").join("agents"),
                DirType::AgentsHome => base.join(".agents"),
            }
        } else {
            // NOTE: "Block" is kept here for backwards compatibility with existing
            // user config/data directories (e.g. ~/Library/Application Support/Block/goose/).
            // Changing this would orphan existing installations.
            let strategy = choose_app_strategy(AppStrategyArgs {
                top_level_domain: "Block".to_string(),
                author: "Block".to_string(),
                app_name: "goose".to_string(),
            })
            .expect("goose requires a home dir");

            match dir_type {
                DirType::Config => strategy.config_dir(),
                DirType::Data => strategy.data_dir(),
                DirType::State => strategy.state_dir().unwrap_or(strategy.data_dir()),
                DirType::Plugins => strategy.home_dir().join(".agents").join("plugins"),
                DirType::Agents => strategy.home_dir().join(".agents").join("agents"),
                DirType::AgentsHome => strategy.home_dir().join(".agents"),
            }
        }
    }

    pub(crate) fn path_root() -> Option<PathBuf> {
        Self::validated_path_root(std::env::var_os("GOOSE_PATH_ROOT"))
    }

    fn validated_path_root(value: Option<OsString>) -> Option<PathBuf> {
        value.map(PathBuf::from).filter(|path| path.is_absolute())
    }

    pub fn config_dir() -> PathBuf {
        Self::get_dir(DirType::Config)
    }

    pub fn data_dir() -> PathBuf {
        Self::get_dir(DirType::Data)
    }

    pub fn state_dir() -> PathBuf {
        Self::get_dir(DirType::State)
    }

    pub fn plugins_dir() -> PathBuf {
        Self::get_dir(DirType::Plugins)
    }

    pub fn agents_dir() -> PathBuf {
        Self::get_dir(DirType::Agents)
    }

    pub fn agents_home_dir() -> PathBuf {
        Self::get_dir(DirType::AgentsHome)
    }

    pub fn in_agents_home_dir(subpath: &str) -> PathBuf {
        Self::agents_home_dir().join(subpath)
    }

    pub fn in_state_dir(subpath: &str) -> PathBuf {
        Self::state_dir().join(subpath)
    }

    pub fn in_config_dir(subpath: &str) -> PathBuf {
        Self::config_dir().join(subpath)
    }

    pub fn in_data_dir(subpath: &str) -> PathBuf {
        Self::data_dir().join(subpath)
    }

    /// The Goose App writes what Claude Code sets per project folder (plugin skills,
    /// commands, plugin agents, hooks, skill listing rules) for each folder a code chat
    /// works in to `<config>/ordner/<key>` (`server/code-chats.mjs` › `codeOrdnerSync`),
    /// laid out like the profile itself. `None` when there is none for this folder; the
    /// caller then reads the profile's own copies, written for the process's folder.
    pub fn folder_config_dir(working_dir: &Path) -> Option<PathBuf> {
        let dir = Self::config_dir()
            .join("ordner")
            .join(folder_key(working_dir));
        dir.is_dir().then_some(dir)
    }

    /// `folder_config_dir`, else the profile's config folder.
    pub fn config_dir_for(working_dir: Option<&Path>) -> PathBuf {
        working_dir
            .and_then(Self::folder_config_dir)
            .unwrap_or_else(Self::config_dir)
    }
}

/// Claude Code's project folder name (every character that is not an ASCII letter or digit
/// becomes `-`, one per UTF-16 unit as in JavaScript), lower case so that the drive letter's
/// case does not matter, trailing separators dropped. Must match `server/code-chats.mjs` ›
/// `codeOrdnerKey`.
pub fn folder_key(working_dir: &Path) -> String {
    let text = working_dir.to_string_lossy();
    let trimmed = text.trim_end_matches(['\\', '/']);
    let trimmed = if trimmed.is_empty() {
        text.as_ref()
    } else {
        trimmed
    };
    let mut key = String::with_capacity(trimmed.len());
    for c in trimmed.chars() {
        if c.is_ascii_alphanumeric() {
            key.push(c.to_ascii_lowercase());
        } else {
            key.extend(std::iter::repeat_n('-', c.len_utf16()));
        }
    }
    key
}

enum DirType {
    Config,
    Data,
    State,
    Plugins,
    Agents,
    AgentsHome,
}

#[cfg(test)]
mod tests {
    use super::{folder_key, Paths};
    use std::ffi::OsString;
    use std::path::Path;

    #[test]
    fn folder_key_matches_the_app_server() {
        assert_eq!(
            folder_key(Path::new(r"C:\Users\anton\Goose-App\goose-native\")),
            "c--users-anton-goose-app-goose-native"
        );
        assert_eq!(folder_key(Path::new("c:/Pers\u{f6}nlich")), "c--pers-nlich");
    }

    #[test]
    fn path_root_requires_an_absolute_path() {
        assert_eq!(Paths::validated_path_root(None), None);
        assert_eq!(Paths::validated_path_root(Some(OsString::new())), None);
        assert_eq!(
            Paths::validated_path_root(Some(OsString::from("relative/root"))),
            None
        );

        let absolute = std::env::current_dir()
            .unwrap()
            .join("nonexistent-goose-root");
        assert_eq!(
            Paths::validated_path_root(Some(absolute.clone().into_os_string())),
            Some(absolute)
        );
    }
}
