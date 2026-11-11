use crate::config::paths::Paths;
use include_dir::{include_dir, Dir};
use minijinja::{Environment, Error as MiniJinjaError, Value as MJValue};
use serde::Serialize;
use std::path::{Path, PathBuf};
use tracing::warn;

/// Folder with prompt templates shared by several goose processes that have separate config
/// folders. A file `<folder>/<name>` goes before `<config>/prompts/<name>` and the built-in text.
pub const PROMPTS_DIR_ENV: &str = "GOOSE_PROMPTS_DIR";

static CORE_PROMPTS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/src/prompts");

static TEMPLATE_REGISTRY: &[(&str, &str)] = &[
    (
        "system.md",
        "Main system prompt that defines goose's personality and behavior",
    ),
    (
        "claude_system.md",
        "Main system prompt for Claude and all models outside the OpenAI GPT family, with Claude Code's working rules",
    ),
    (
        "openai_system.md",
        "Main system prompt for OpenAI GPT models, based on OpenAI's Codex instructions",
    ),
    (
        "compaction.md",
        "Prompt for summarizing conversation history when context limits are reached",
    ),
    (
        "compaction_summary.md",
        "Renders the structured compaction output into the post-compaction context",
    ),
    (
        "subagent_system.md",
        "System prompt for subagents spawned to handle specific tasks",
    ),
    (
        "apps_create.md",
        "Prompt for generating new Goose apps based on the user instructions",
    ),
    (
        "apps_iterate.md",
        "Prompt for updating existing Goose apps based on feedback",
    ),
    (
        "permission_judge.md",
        "Prompt for analyzing tool operations for read-only detection",
    ),
    (
        "tiny_model_system.md",
        "System prompt for tiny local models using shell command emulation",
    ),
    (
        "session_name.md",
        "System prompt for generating short session names from conversation history",
    ),
];

/// Information about a template including its content and customization status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Template {
    pub name: String,
    pub description: String,
    pub default_content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_content: Option<String>,
    pub is_customized: bool,
}

fn builtin_content(name: &str) -> Option<String> {
    CORE_PROMPTS_DIR
        .get_file(name)
        .map(|file| String::from_utf8_lossy(file.contents()).to_string())
        .or_else(|| goose_context_management::templates::builtin_template(name))
}

fn user_prompts_dir() -> PathBuf {
    Paths::config_dir().join("prompts")
}

fn shared_prompts_dir() -> Option<PathBuf> {
    std::env::var_os(PROMPTS_DIR_ENV)
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
}

/// Text of a template file without byte order mark and with LF line ends. A missing, unreadable
/// or empty file gives `None`, so the next source applies instead of an error or an empty prompt.
fn read_template_file(path: &Path) -> Option<String> {
    if !path.exists() {
        return None;
    }
    match std::fs::read_to_string(path) {
        Ok(raw) => {
            let text = raw
                .strip_prefix('\u{feff}')
                .unwrap_or(&raw)
                .replace("\r\n", "\n");
            if text.trim().is_empty() {
                warn!(file = %path.display(), "prompt template: empty file, next source applies");
                None
            } else {
                Some(text)
            }
        }
        Err(error) => {
            warn!(file = %path.display(), %error, "prompt template: file not readable, next source applies");
            None
        }
    }
}

fn custom_content_in(dirs: &[PathBuf], name: &str) -> Option<String> {
    dirs.iter()
        .find_map(|dir| read_template_file(&dir.join(name)))
}

/// The template from a file: the shared folder first, then the config folder of this process.
fn custom_content(name: &str) -> Option<String> {
    let dirs: Vec<PathBuf> = shared_prompts_dir()
        .into_iter()
        .chain(std::iter::once(user_prompts_dir()))
        .collect();
    custom_content_in(&dirs, name)
}

fn builtin_or_error(name: &str) -> Result<String, MiniJinjaError> {
    builtin_content(name).ok_or_else(|| {
        MiniJinjaError::new(
            minijinja::ErrorKind::TemplateNotFound,
            format!("Built-in template '{}' not found", name),
        )
    })
}

/// Renders the template from a file when there is one; a file that does not render falls back
/// to the built-in text with a warning, so one broken file cannot stop every chat.
fn render_with_fallback<T: Serialize>(
    name: &str,
    custom: Option<String>,
    context: &T,
) -> Result<String, MiniJinjaError> {
    if let Some(template_str) = custom {
        match render_string(&template_str, context) {
            Ok(rendered) => return Ok(rendered),
            Err(error) => {
                warn!(template = name, %error, "prompt template: file does not render, built-in text applies")
            }
        }
    }
    render_string(&builtin_or_error(name)?, context)
}

fn is_registered(name: &str) -> bool {
    TEMPLATE_REGISTRY.iter().any(|(n, _)| *n == name)
}

/// Wrap code in a markdown fence longer than any backtick run it contains,
/// so embedded fences cannot close the block early.
fn code_fence(code: String) -> String {
    let longest_run = code
        .chars()
        .fold((0usize, 0usize), |(max, run), c| {
            if c == '`' {
                (max.max(run + 1), run + 1)
            } else {
                (max, 0)
            }
        })
        .0;
    let fence = "`".repeat((longest_run + 1).max(3));
    format!("{fence}\n{}\n{fence}", code.trim_end_matches('\n'))
}

pub fn render_string<T: Serialize>(
    template_str: &str,
    context: &T,
) -> Result<String, MiniJinjaError> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_filter("code_fence", code_fence);
    env.add_template("template", template_str)?;
    let tmpl = env.get_template("template")?;
    let ctx = MJValue::from_serialize(context);
    let rendered = tmpl.render(ctx)?;
    Ok(rendered.trim().to_string())
}

pub fn render_template<T: Serialize>(name: &str, context: &T) -> Result<String, MiniJinjaError> {
    if !is_registered(name) {
        return Err(MiniJinjaError::new(
            minijinja::ErrorKind::TemplateNotFound,
            format!("Template '{}' is not registered", name),
        ));
    }

    render_with_fallback(name, custom_content(name), context)
}

pub fn template_source(name: &str) -> Result<String, MiniJinjaError> {
    match custom_content(name) {
        Some(text) => Ok(text),
        None => builtin_or_error(name),
    }
}

pub fn get_template(name: &str) -> Option<Template> {
    let (_, description) = TEMPLATE_REGISTRY.iter().find(|(n, _)| *n == name)?;

    let default_content = builtin_content(name)?;

    let user_content = custom_content(name);

    let is_customized = user_content.is_some();

    Some(Template {
        name: name.to_string(),
        description: description.to_string(),
        default_content,
        user_content,
        is_customized,
    })
}

pub fn save_template(name: &str, content: &str) -> std::io::Result<()> {
    if !is_registered(name) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Template '{}' is not registered", name),
        ));
    }

    let prompts_dir = user_prompts_dir();
    std::fs::create_dir_all(&prompts_dir)?;
    let path = prompts_dir.join(name);
    std::fs::write(path, content)
}

/// Reset a template to its default by removing the user customization.
pub fn reset_template(name: &str) -> std::io::Result<()> {
    if !is_registered(name) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Template '{}' is not registered", name),
        ));
    }

    let path = user_prompts_dir().join(name);
    if path.exists() {
        std::fs::remove_file(path)
    } else {
        Ok(())
    }
}

pub fn list_templates() -> Vec<Template> {
    TEMPLATE_REGISTRY
        .iter()
        .filter_map(|(name, description)| {
            let default_content = builtin_content(name)?;

            let user_content = custom_content(name);

            let is_customized = user_content.is_some();

            Some(Template {
                name: name.to_string(),
                description: description.to_string(),
                default_content,
                user_content,
                is_customized,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_get_template() {
        let template = get_template("system.md");
        assert!(template.is_some(), "system.md should be registered");

        let template = template.unwrap();
        assert_eq!(template.name, "system.md");
        assert!(!template.description.is_empty());
        assert!(!template.default_content.is_empty());
        assert!(!template.is_customized);
    }

    #[test]
    fn test_render_template() {
        let context: HashMap<String, String> = HashMap::new();
        let result = render_template("system.md", &context);
        assert!(result.is_ok(), "Should be able to render system.md");
        assert!(!result.unwrap().is_empty());
    }

    #[test]
    fn shared_folder_goes_before_the_config_folder() {
        let shared = tempfile::tempdir().unwrap();
        let config = tempfile::tempdir().unwrap();
        std::fs::write(shared.path().join("a.md"), "\u{feff}shared\r\ntext\r\n").unwrap();
        std::fs::write(config.path().join("a.md"), "config").unwrap();
        std::fs::write(config.path().join("b.md"), "config only").unwrap();
        let dirs = [shared.path().to_path_buf(), config.path().to_path_buf()];
        assert_eq!(
            custom_content_in(&dirs, "a.md").as_deref(),
            Some("shared\ntext\n")
        );
        assert_eq!(
            custom_content_in(&dirs, "b.md").as_deref(),
            Some("config only")
        );
        assert_eq!(custom_content_in(&dirs, "c.md"), None);
    }

    #[test]
    fn unusable_file_leaves_the_next_source_in_place() {
        let shared = tempfile::tempdir().unwrap();
        let config = tempfile::tempdir().unwrap();
        std::fs::write(shared.path().join("empty.md"), " \r\n").unwrap();
        std::fs::write(shared.path().join("binary.md"), [0xff, 0xfe, 0x00, 0xd8]).unwrap();
        std::fs::write(config.path().join("empty.md"), "config").unwrap();
        let dirs = [shared.path().to_path_buf(), config.path().to_path_buf()];
        assert_eq!(
            custom_content_in(&dirs, "empty.md").as_deref(),
            Some("config")
        );
        assert_eq!(custom_content_in(&dirs, "binary.md"), None);
    }

    #[test]
    fn file_that_does_not_render_falls_back_to_the_built_in_text() {
        let context: HashMap<String, String> = HashMap::new();
        let built_in = render_with_fallback("system.md", None, &context).unwrap();
        let broken = Some("{% if %} never closed".to_string());
        assert_eq!(
            render_with_fallback("system.md", broken, &context).unwrap(),
            built_in
        );
        let custom = Some("from file".to_string());
        assert_eq!(
            render_with_fallback("system.md", custom, &context).unwrap(),
            "from file"
        );
    }

    #[test]
    fn test_list_templates() {
        let templates = list_templates();
        assert_eq!(templates.len(), TEMPLATE_REGISTRY.len());

        let has_system = templates.iter().any(|t| t.name == "system.md");
        assert!(has_system, "system.md should be in the template list");

        for template in templates {
            assert!(
                !template.description.is_empty(),
                "Each template should have a description"
            );
            assert!(
                !template.default_content.is_empty(),
                "Each template should have content"
            );
        }
    }
}
