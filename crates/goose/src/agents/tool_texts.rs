//! Tool descriptions from files instead of the build.
//!
//! With `GOOSE_TOOL_TEXTS_DIR` set to a folder, a file `<folder>/<tool name>.md` replaces the
//! description of that tool, for tools of extensions that are part of goose itself (platform
//! extensions) and for `tool_search`. The file name is the tool name the model sees, e.g.
//! `read.md` or `todo__todo_write.md`. A byte order mark is dropped, CRLF becomes LF and one
//! line break at the end of the file is dropped; everything else is taken word for word.
//!
//! The folder is read once per process, so a changed text takes effect after a restart. A tool
//! without a file, an unreadable or empty file, a missing folder or no variable at all leave the
//! built-in description in place, so goose behaves as without this module.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use rmcp::model::Tool;
use tracing::warn;

pub const TOOL_TEXTS_DIR_ENV: &str = "GOOSE_TOOL_TEXTS_DIR";

static TEXTS: OnceLock<HashMap<String, String>> = OnceLock::new();

fn texts() -> &'static HashMap<String, String> {
    TEXTS.get_or_init(|| match std::env::var_os(TOOL_TEXTS_DIR_ENV) {
        Some(dir) if !dir.is_empty() => load(Path::new(&dir)),
        _ => HashMap::new(),
    })
}

/// Reads every `<name>.md` in `dir`; files that cannot be used are left out with a warning.
pub(crate) fn load(dir: &Path) -> HashMap<String, String> {
    let mut texts = HashMap::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            warn!(dir = %dir.display(), %error, "tool texts: folder not readable, built-in descriptions apply");
            return texts;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("md") {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        match std::fs::read_to_string(&path) {
            Ok(raw) => match normalize(&raw) {
                Some(text) => {
                    texts.insert(name.to_string(), text);
                }
                None => {
                    warn!(file = %path.display(), "tool texts: empty file, built-in description applies")
                }
            },
            Err(error) => {
                warn!(file = %path.display(), %error, "tool texts: file not readable, built-in description applies")
            }
        }
    }
    texts
}

fn normalize(raw: &str) -> Option<String> {
    let text = raw
        .strip_prefix('\u{feff}')
        .unwrap_or(raw)
        .replace("\r\n", "\n");
    let text = text.strip_suffix('\n').unwrap_or(&text);
    if text.trim().is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn apply_from(texts: &HashMap<String, String>, tool: &mut Tool) {
    if let Some(text) = texts.get(tool.name.as_ref()) {
        tool.description = Some(text.clone().into());
    }
}

/// Replaces the description of `tool` when the folder holds a text for it.
pub(crate) fn apply(tool: &mut Tool) {
    apply_from(texts(), tool);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;
    use std::sync::Arc;

    fn tool(name: &str, description: &str) -> Tool {
        Tool::new(
            name.to_string(),
            description.to_string(),
            Arc::new(Map::new()),
        )
    }

    #[test]
    fn file_text_replaces_the_description_word_for_word() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("read.md"),
            "\u{feff}Line one.\r\n\r\nLine two.\r\n",
        )
        .unwrap();
        let texts = load(dir.path());
        let mut read = tool("read", "built-in");
        apply_from(&texts, &mut read);
        assert_eq!(read.description.as_deref(), Some("Line one.\n\nLine two."));
    }

    #[test]
    fn missing_empty_or_other_files_keep_the_built_in_text() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("edit.md"), " \n").unwrap();
        std::fs::write(dir.path().join("write.txt"), "not a tool text").unwrap();
        let texts = load(dir.path());
        for name in ["edit", "write", "glob"] {
            let mut t = tool(name, "built-in");
            apply_from(&texts, &mut t);
            assert_eq!(t.description.as_deref(), Some("built-in"), "{name}");
        }
        assert!(load(&dir.path().join("does-not-exist")).is_empty());
    }

    #[test]
    fn only_one_trailing_line_break_is_dropped() {
        assert_eq!(normalize("text\n\n").as_deref(), Some("text\n"));
        assert_eq!(normalize("text").as_deref(), Some("text"));
    }
}
