//! Everything specific to skills: filesystem discovery (`SKILL.md` walking +
//! built-ins) and the runtime MCP client (`client` submodule). User-facing
//! CRUD lives in `crate::sources`, which generalizes across source types.

mod builtin;
pub mod client;
mod supporting_files;

pub use client::{SkillsClient, EXTENSION_NAME};
pub(crate) use supporting_files::{
    create_source_file, load_supporting_file, read_source_file, write_source_file,
};

use crate::config::{paths::Paths, Config};
use crate::plugins::discovery::PluginScope;
use crate::plugins::{
    configured_project_plugin_skill_dirs, enabled_plugin_skill_dirs_with_config,
    installed_plugin_skill_dirs,
};
use crate::sources::parse_frontmatter;
use agent_client_protocol::Error;
use anyhow::Result;
use goose_agent::skills::{SkillNaming, SkillRoot};
use goose_sdk_types::custom_requests::{SourceEntry, SourceType};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tracing::warn;

#[derive(Debug, Deserialize)]
pub struct SkillFrontmatter {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: String,
    /// Free-form bag for caller-defined fields. Per the agentskills.io spec
    /// (<https://agentskills.io/specification#frontmatter>), arbitrary
    /// metadata lives in this nested mapping so it doesn't collide with
    /// reserved frontmatter fields.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// Canonical writable location for global user skills: `~/.agents/skills`.
pub fn global_skills_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".agents").join("skills"))
}

/// Canonical writable location for project-scoped skills:
/// `<project>/.agents/skills`.
pub fn project_skills_dir(project_dir: &Path) -> PathBuf {
    project_dir.join(".agents").join("skills")
}

pub(crate) fn skills_dir_global_or_err() -> Result<PathBuf, Error> {
    global_skills_dir()
        .ok_or_else(|| Error::internal_error().data("Could not determine home directory"))
}

pub(crate) fn skills_dir_project_or_err(project_dir: &str) -> Result<PathBuf, Error> {
    if project_dir.trim().is_empty() {
        return Err(
            Error::invalid_params().data("projectDir must not be empty when global is false")
        );
    }
    Ok(project_skills_dir(Path::new(project_dir)))
}

pub(crate) fn skill_base_dir(global: bool, project_dir: Option<&str>) -> Result<PathBuf, Error> {
    if global {
        skills_dir_global_or_err()
    } else {
        let pd = project_dir.ok_or_else(|| {
            Error::invalid_params().data("projectDir is required when global is false")
        })?;
        skills_dir_project_or_err(pd)
    }
}

pub(crate) fn validate_skill_name(name: &str) -> Result<(), Error> {
    if name.is_empty() {
        return Err(Error::invalid_params().data("Skill name must not be empty"));
    }
    if name.len() > 64 {
        return Err(Error::invalid_params().data(format!(
            "Invalid skill name \"{}\". Names must be at most 64 characters.",
            name
        )));
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
    {
        return Err(Error::invalid_params().data(format!(
            "Invalid skill name \"{}\". Names may only contain lowercase letters, digits, and hyphens.",
            name
        )));
    }
    if name.starts_with('-') || name.ends_with('-') {
        return Err(Error::invalid_params().data(format!(
            "Invalid skill name \"{}\". Names must not start or end with a hyphen.",
            name
        )));
    }
    Ok(())
}

const DEFAULT_GOOSE_DOCS_ROOT: &str = "https://goose-docs.ai";
const GOOSE_DOCS_ROOT_PLACEHOLDER: &str = "{{GOOSE_DOCS_ROOT}}";

/// Substitute the `{{GOOSE_DOCS_ROOT}}` placeholder in the builtin
/// `goose-doc-guide` skill with the resolved docs root. Resolution is
/// deterministic: the configured `GOOSE_DOCS_ROOT` if set, otherwise the
/// canonical online docs root.
fn resolve_docs_root_placeholder(skill: &SourceEntry, content: &str, docs_root: &str) -> String {
    if skill.name != "goose-doc-guide" || skill.source_type != SourceType::BuiltinSkill {
        return content.to_string();
    }

    content.replace(GOOSE_DOCS_ROOT_PLACEHOLDER, docs_root)
}

pub fn loaded_skill_context_with_args(skill: &SourceEntry, args: Option<&str>) -> Result<String> {
    let content = goose_agent::skills::skill_content_with_args(skill, args)?;
    let docs_root = Config::global()
        .get_goose_docs_root()?
        .unwrap_or_else(|| DEFAULT_GOOSE_DOCS_ROOT.to_string());
    let content = resolve_docs_root_placeholder(skill, &content, &docs_root);

    Ok(goose_agent::skills::loaded_skill_context(skill, &content))
}

pub fn skill_argument_hint(skill: &SourceEntry) -> Option<String> {
    skill
        .properties
        .get("argument-hint")
        .and_then(|value| value.as_str())
        .filter(|hint| !hint.is_empty())
        .map(str::to_string)
}

fn canonicalize_or_original(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn is_lexical_project_plugin_path(path: &Path) -> bool {
    if path.starts_with(Paths::plugins_dir()) {
        return false;
    }

    path.ancestors().any(|ancestor| {
        ancestor.file_name().and_then(|name| name.to_str()) == Some("plugins")
            && ancestor
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                == Some(".agents")
    })
}

fn inferred_discoverable_skill_root(path: &Path) -> Option<PathBuf> {
    let canonical_path = canonicalize_or_original(path);

    let mut global_roots = Vec::new();
    if let Some(global_root) = global_skills_dir() {
        global_roots.push(global_root);
    }
    global_roots.push(Paths::config_dir().join("skills"));
    if let Some(home) = dirs::home_dir() {
        global_roots.push(home.join(".claude").join("skills"));
        global_roots.push(home.join(".config").join("agents").join("skills"));
    }
    global_roots.extend(installed_plugin_skill_dirs());

    for root in global_roots {
        let canonical_root = canonicalize_or_original(&root);
        if canonical_path.starts_with(&canonical_root) {
            return Some(canonical_root);
        }
    }

    canonical_path.ancestors().find_map(|ancestor| {
        let parent = ancestor.parent()?;
        let is_project_skills_root = ancestor.file_name().and_then(|name| name.to_str())
            == Some("skills")
            && matches!(
                parent.file_name().and_then(|name| name.to_str()),
                Some(".goose") | Some(".claude") | Some(".agents")
            );
        is_project_skills_root.then(|| ancestor.to_path_buf())
    })
}

fn resolve_discoverable_skill_dir_with_config(
    path: &str,
    config: &Config,
) -> Result<PathBuf, Error> {
    if path.is_empty() {
        return Err(Error::invalid_params().data("Source path must not be empty"));
    }

    if is_lexical_project_plugin_path(Path::new(path)) {
        return Err(Error::invalid_params().data(format!("Source \"{}\" not found", path)));
    }

    let canonical_dir = Path::new(path)
        .canonicalize()
        .map_err(|_| Error::invalid_params().data(format!("Source \"{}\" not found", path)))?;

    let configured_project_plugin_path = !Path::new(path).starts_with(Paths::plugins_dir())
        && configured_project_plugin_skill_dirs(config)
            .into_iter()
            .map(|root| canonicalize_or_original(&root))
            .any(|root| canonical_dir.starts_with(root));

    if configured_project_plugin_path
        || inferred_discoverable_skill_root(&canonical_dir).is_none()
        || !canonical_dir.is_dir()
        || !canonical_dir.join("SKILL.md").is_file()
    {
        return Err(Error::invalid_params().data(format!("Source \"{}\" not found", path)));
    }

    Ok(canonical_dir)
}

pub(crate) fn resolve_discoverable_skill_dir(path: &str) -> Result<PathBuf, Error> {
    resolve_discoverable_skill_dir_with_config(path, Config::global())
}

pub(crate) fn resolve_skill_dir(path: &str) -> Result<PathBuf, Error> {
    resolve_discoverable_skill_dir(path)
}

pub(crate) fn is_global_skill_dir(path: &Path) -> bool {
    global_skills_dir().as_deref().is_some_and(|root| {
        canonicalize_or_original(path).starts_with(canonicalize_or_original(root))
    })
}

pub(crate) fn infer_skill_name(dir: &Path) -> String {
    let md = dir.join("SKILL.md");
    if let Ok(raw) = std::fs::read_to_string(&md) {
        if let Ok(Some((meta, _))) = parse_frontmatter::<SkillFrontmatter>(&raw) {
            if let Some(n) = meta.name.filter(|n| !n.is_empty()) {
                return n;
            }
        }
    }
    dir.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unnamed")
        .to_string()
}

pub(crate) fn build_skill_md(
    name: &str,
    description: &str,
    content: &str,
    metadata: &HashMap<String, Value>,
) -> String {
    let safe_desc = description.replace('\'', "''");
    let mut md = String::from("---\n");
    md.push_str(&format!("name: {}\n", name));
    md.push_str(&format!("description: '{}'\n", safe_desc));
    if !metadata.is_empty() {
        md.push_str("metadata:\n");
        // Use YAML for the nested metadata block. We render it with serde_yaml
        // and indent every line by two spaces so it nests under `metadata:`.
        let yaml = serde_yaml::to_string(metadata).unwrap_or_default();
        for line in yaml.lines() {
            if line.is_empty() {
                continue;
            }
            md.push_str("  ");
            md.push_str(line);
            md.push('\n');
        }
    }
    md.push_str("---\n");
    if !content.is_empty() {
        md.push('\n');
        md.push_str(content);
        md.push('\n');
    }
    md
}

pub(crate) fn parse_skill_frontmatter(raw: &str) -> (String, String) {
    if !raw.trim_start().starts_with("---") {
        return (String::new(), raw.to_string());
    }
    match parse_frontmatter::<SkillFrontmatter>(raw) {
        Ok(Some((meta, body))) => (meta.description, body),
        _ => (String::new(), raw.to_string()),
    }
}

/// Every directory the agent reads skills from, paired with whether each is a
/// global (home-rooted) location. Order matches discovery precedence: project
/// dirs first, then global dirs.
pub fn all_skill_dirs(working_dir: Option<&Path>) -> Vec<(PathBuf, bool)> {
    all_skill_dirs_with_config(working_dir, Config::global())
        .into_iter()
        .map(|dir| (dir.path, dir.is_global))
        .collect()
}

struct SkillDirectory {
    path: PathBuf,
    is_global: bool,
    writable: bool,
    preserve_path: bool,
    /// Search subdirectories for `SKILL.md`; `false` reads only the `SKILL.md` of
    /// each child directory, like Claude Code reads its skill folders.
    recursive: bool,
    naming: SkillNaming,
}

/// `GOOSE_SKILL_LISTING` marks the Goose App's Claude Code profile: the server fills it
/// from Claude Code's settings and sets it only for Goose Code (`server/code-chats.mjs`
/// › `claudeSetup`, `buildCodeEnv`). Goose Code then reads skills where Claude Code does.
fn claude_code_profile() -> bool {
    std::env::var_os("GOOSE_SKILL_LISTING").is_some()
}

fn all_skill_dirs_with_config(working_dir: Option<&Path>, config: &Config) -> Vec<SkillDirectory> {
    skill_dirs_with_config(
        claude_code_profile(),
        dirs::home_dir().as_deref(),
        working_dir,
        config,
    )
}

fn skill_dirs_with_config(
    claude_profile: bool,
    home: Option<&Path>,
    working_dir: Option<&Path>,
    config: &Config,
) -> Vec<SkillDirectory> {
    if claude_profile {
        return claude_skill_dirs(home, working_dir, config);
    }
    goose_skill_dirs(home, working_dir, config)
}

/// Claude Code's own skill locations (Claude Code 2.1.319: `pWr` in its program; docs
/// code.claude.com/docs/en/skills), in Claude's order: the personal `~/.claude/skills`,
/// the project's `.claude/skills` folders, then the plugin skills and the commands.
/// Every folder holds skills one level down, named after their folder; subfolders are
/// not searched, `~/.agents` is no Claude location, and the reserved child name `synced`
/// never becomes a skill. The account's synced skills arrive as the plugin
/// `anthropic-skills`, the name Claude gives them, because only the App server knows
/// who is signed in (`server/code-chats.mjs` › `syncPlugins`).
fn claude_skill_dirs(
    home: Option<&Path>,
    working_dir: Option<&Path>,
    config: &Config,
) -> Vec<SkillDirectory> {
    let mut dirs = Vec::new();
    if let Some(h) = home {
        dirs.push(SkillDirectory {
            path: h.join(".claude").join("skills"),
            is_global: true,
            writable: true,
            preserve_path: false,
            recursive: false,
            naming: SkillNaming::Folder,
        });
    }

    if let Some(wd) = working_dir {
        for path in project_skill_dirs(wd, home) {
            dirs.push(SkillDirectory {
                path,
                is_global: false,
                writable: true,
                preserve_path: false,
                recursive: false,
                naming: SkillNaming::Folder,
            });
        }
    }

    // Plugin skills and commands, as Claude Code shows them: skills named
    // `<plugin>:<skill>`, commands by the `name` the App server writes, which a folder
    // cannot hold on Windows. Both depend on the project folder (its enabled plugins and
    // commands); the App server writes them per chat folder (`server/code-chats.mjs` ›
    // `syncPlugins`, `codeOrdnerSync`).
    let plugin_config = Paths::config_dir_for(working_dir);
    for (path, plugin) in claude_plugin_skill_dirs(&plugin_config) {
        dirs.push(SkillDirectory {
            path,
            is_global: true,
            writable: true,
            preserve_path: false,
            recursive: false,
            naming: SkillNaming::PluginFolder(plugin),
        });
    }
    dirs.push(SkillDirectory {
        path: plugin_config.join("skills").join("befehle"),
        is_global: true,
        writable: true,
        preserve_path: false,
        recursive: false,
        naming: SkillNaming::Frontmatter,
    });

    // Goose's own plugins are no Claude location; they stay as in the normal profile.
    for (path, scope) in enabled_plugin_skill_dirs_with_config(working_dir, config) {
        dirs.push(SkillDirectory {
            path,
            is_global: scope == PluginScope::User,
            writable: scope == PluginScope::User,
            preserve_path: true,
            recursive: true,
            naming: SkillNaming::Frontmatter,
        });
    }
    dirs
}

/// Claude Code's project skill folders: every directory at or above the working
/// directory that has `.claude/skills`, nearest first. The walk ends at the enclosing
/// git root, or without a repository at the home directory, which never counts as a
/// project folder (Claude Code 2.1.318: `Xae`).
fn project_skill_dirs(working_dir: &Path, home: Option<&Path>) -> Vec<PathBuf> {
    let start = working_dir
        .canonicalize()
        .unwrap_or_else(|_| working_dir.to_path_buf());
    let home = home.map(|path| path.canonicalize().unwrap_or_else(|_| path.to_path_buf()));
    let mut dirs = Vec::new();
    let mut current = Some(start.as_path());
    while let Some(dir) = current {
        if Some(dir) == home.as_deref() {
            break;
        }
        let candidate = dir.join(".claude").join("skills");
        if candidate.is_dir() {
            dirs.push(candidate);
        }
        if dir.join(".git").exists() {
            break;
        }
        current = dir.parent();
    }
    dirs
}

/// Skill folders of the plugins Claude Code has installed; the App links each plugin's
/// `skills` folder here under the plugin's name (`server/code-chats.mjs` › `syncPlugins`).
fn claude_plugin_skill_dirs(config_dir: &Path) -> Vec<(PathBuf, String)> {
    let plugins = config_dir.join("skills").join("plugins");
    let Ok(entries) = std::fs::read_dir(plugins) else {
        return Vec::new();
    };
    let mut dirs: Vec<(PathBuf, String)> = entries
        .flatten()
        .filter_map(|entry| {
            let plugin = entry.file_name().to_str()?.to_string();
            let path = entry.path();
            path.is_dir().then_some((path, plugin))
        })
        .collect();
    dirs.sort_by(|a, b| a.1.cmp(&b.1));
    dirs
}

fn goose_skill_dirs(
    home: Option<&Path>,
    working_dir: Option<&Path>,
    config: &Config,
) -> Vec<SkillDirectory> {
    let mut dirs = Vec::new();
    let plugin_dirs = enabled_plugin_skill_dirs_with_config(working_dir, config);

    if let Some(wd) = working_dir {
        for path in [
            wd.join(".agents").join("skills"),
            wd.join(".goose").join("skills"),
            wd.join(".claude").join("skills"),
        ] {
            dirs.push(SkillDirectory {
                path,
                is_global: false,
                writable: true,
                preserve_path: false,
                recursive: true,
                naming: SkillNaming::Frontmatter,
            });
        }
    }
    dirs.extend(
        plugin_dirs
            .iter()
            .filter(|(_, scope)| *scope == PluginScope::Project)
            .map(|(path, _)| SkillDirectory {
                path: path.clone(),
                is_global: false,
                writable: false,
                preserve_path: true,
                recursive: true,
                naming: SkillNaming::Frontmatter,
            }),
    );

    if let Some(h) = home.as_ref() {
        dirs.push(SkillDirectory {
            path: h.join(".agents").join("skills"),
            is_global: true,
            writable: true,
            preserve_path: false,
            recursive: true,
            naming: SkillNaming::Frontmatter,
        });
    }
    dirs.push(SkillDirectory {
        path: Paths::config_dir().join("skills"),
        is_global: true,
        writable: true,
        preserve_path: false,
        recursive: true,
        naming: SkillNaming::Frontmatter,
    });
    if let Some(h) = home.as_ref() {
        for path in [
            h.join(".claude").join("skills"),
            h.join(".config").join("agents").join("skills"),
        ] {
            dirs.push(SkillDirectory {
                path,
                is_global: true,
                writable: true,
                preserve_path: false,
                recursive: true,
                naming: SkillNaming::Frontmatter,
            });
        }
    }

    dirs.extend(
        plugin_dirs
            .into_iter()
            .filter(|(_, scope)| *scope == PluginScope::User)
            .map(|(path, _)| SkillDirectory {
                path,
                is_global: true,
                writable: true,
                preserve_path: true,
                recursive: true,
                naming: SkillNaming::Frontmatter,
            }),
    );

    dirs
}

fn parse_skill_content(
    content: &str,
    path: &Path,
    global: bool,
    writable: bool,
) -> Option<SourceEntry> {
    let (metadata, body): (SkillFrontmatter, String) = match parse_frontmatter(content) {
        Ok(Some(parsed)) => parsed,
        Ok(None) => return None,
        Err(e) => {
            warn!("Failed to parse skill frontmatter: {}", e);
            return None;
        }
    };

    let name = match metadata.name.filter(|n| !n.is_empty()) {
        Some(n) => n,
        None => {
            warn!(
                "Skill at '{}' is missing a required 'name' in frontmatter, skipping",
                path.display()
            );
            return None;
        }
    };

    if name.contains('/') {
        warn!("Skill name '{}' contains '/', skipping", name);
        return None;
    }

    Some(SourceEntry {
        source_type: SourceType::Skill,
        name,
        description: metadata.description,
        content: body,
        path: path.to_string_lossy().into_owned(),
        global,
        writable,
        supporting_files: Vec::new(),
        properties: metadata.metadata,
    })
}

/// Discover skills from all configured filesystem locations and built-ins.
/// Each returned entry has `global` set according to the directory it was
/// found in (or `true` for built-ins).
/// The skill called `name`. As in Claude Code, a bare name also reaches a namespaced
/// skill (`pdf` reaches `anthropic-skills:pdf`) if exactly one answers to it.
pub fn find_skill<'a>(skills: &'a [SourceEntry], name: &str) -> Option<&'a SourceEntry> {
    if let Some(skill) = skills.iter().find(|skill| skill.name == name) {
        return Some(skill);
    }
    if name.contains(':') {
        return None;
    }
    let suffix = format!(":{name}");
    let mut namespaced = skills.iter().filter(|skill| skill.name.ends_with(&suffix));
    let only = namespaced.next()?;
    namespaced.next().is_none().then_some(only)
}

pub fn discover_skills(working_dir: Option<&Path>) -> Vec<SourceEntry> {
    discover_skills_with_config(working_dir, Config::global())
}

/// Every skill, switched off or not, for listings that let the user switch them
/// (`sources/list`). A switched-off skill carries the property `disabled: true`.
pub fn discover_all_skills(working_dir: Option<&Path>) -> Vec<SourceEntry> {
    let config = Config::global();
    let disabled = disabled_skills(config);
    let mut skills = discover_all_skills_with_config(working_dir, config);
    for skill in &mut skills {
        if disabled.contains(&skill.name) {
            skill
                .properties
                .insert("disabled".into(), Value::Bool(true));
        }
    }
    skills
}

/// Names of switched-off skills, user-made and built-in alike (config or environment,
/// JSON list): the Goose App's settings write it for Toni's Goose (N749), Goose Code
/// sets it in its environment (C97). A switched-off skill is missing from the system
/// prompt, `load_skill` and the slash commands.
fn disabled_skills(config: &Config) -> HashSet<String> {
    config
        .get_param::<Vec<String>>("GOOSE_DISABLED_SKILLS")
        .unwrap_or_default()
        .into_iter()
        .collect()
}

fn discover_skills_in_dirs(dirs: Vec<SkillDirectory>) -> Vec<SourceEntry> {
    let roots = dirs.into_iter().map(|dir| SkillRoot {
        path: dir.path,
        recursive: dir.recursive,
        naming: dir.naming,
        global: dir.is_global,
        writable: dir.writable,
        preserve_path: dir.preserve_path,
    });
    goose_agent::skills::discover_skills(roots)
}

fn discover_skills_with_config(working_dir: Option<&Path>, config: &Config) -> Vec<SourceEntry> {
    let disabled = disabled_skills(config);
    discover_all_skills_with_config(working_dir, config)
        .into_iter()
        .filter(|skill| !disabled.contains(&skill.name))
        .collect()
}

fn discover_all_skills_with_config(
    working_dir: Option<&Path>,
    config: &Config,
) -> Vec<SourceEntry> {
    let dirs = all_skill_dirs_with_config(working_dir, config);
    let mut skills = discover_skills_in_dirs(dirs);
    let mut seen: HashSet<String> = skills.iter().map(|source| source.name.clone()).collect();

    for content in builtin::get_all() {
        if let Some(source) = parse_skill_content(content, &PathBuf::new(), true, true) {
            if !seen.contains(&source.name) {
                seen.insert(source.name.clone());
                let path = format!("builtin://skills/{}", source.name);
                skills.push(SourceEntry {
                    source_type: SourceType::BuiltinSkill,
                    path,
                    ..source
                });
            }
        }
    }

    skills
}

pub fn list_installed_skills(working_dir: Option<&Path>) -> Vec<SourceEntry> {
    let fallback;
    let wd = match working_dir {
        Some(p) => Some(p),
        None => {
            fallback = std::env::current_dir().ok();
            fallback.as_deref()
        }
    };
    discover_skills(wd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write_test_skill(dir: &Path, name: &str, body: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: Test skill\n---\n{body}"),
        )
        .unwrap();
    }

    fn assert_path_rejected_by_source_crud(path: &str, name: &str, skill_dir: &Path) {
        let update_err = crate::sources::update_source_with_roots(
            SourceType::Skill,
            path,
            name,
            "updated",
            "updated body",
            crate::sources::UpdateSourceOptions {
                properties: Some(HashMap::new()),
                additional_roots: &[],
            },
        )
        .unwrap_err();
        assert!(format!("{update_err:?}").contains("not found"));

        let delete_err = crate::sources::delete_source(SourceType::Skill, path).unwrap_err();
        assert!(format!("{delete_err:?}").contains("not found"));

        let export_err = crate::sources::export_source(SourceType::Skill, path).unwrap_err();
        assert!(format!("{export_err:?}").contains("not found"));
        assert!(skill_dir.join("SKILL.md").is_file());
    }

    fn assert_read_only_and_rejected_by_source_crud(skill: &SourceEntry, skill_dir: &Path) {
        assert!(!skill.global);
        assert!(!skill.writable);
        assert_path_rejected_by_source_crud(&skill.path, &skill.name, skill_dir);
    }

    fn skill_with_content(content: &str) -> SourceEntry {
        SourceEntry {
            source_type: SourceType::Skill,
            name: "test-skill".to_string(),
            description: "Test skill".to_string(),
            content: content.to_string(),
            path: String::new(),
            global: false,
            writable: true,
            supporting_files: Vec::new(),
            properties: HashMap::from([(
                "arguments".to_string(),
                json!(["component", "from", "to"]),
            )]),
        }
    }

    fn builtin_goose_doc_guide_skill() -> SourceEntry {
        SourceEntry {
            source_type: SourceType::BuiltinSkill,
            name: "goose-doc-guide".to_string(),
            description: "Test docs skill".to_string(),
            content: "Docs root: {{GOOSE_DOCS_ROOT}}.".to_string(),
            path: "builtin://skills/goose-doc-guide".to_string(),
            global: true,
            writable: true,
            supporting_files: Vec::new(),
            properties: HashMap::new(),
        }
    }

    #[test]
    fn loaded_skill_context_with_args_replaces_arguments_placeholder_with_raw_args() {
        let skill = skill_with_content("Review $ARGUMENTS carefully.");

        let rendered = loaded_skill_context_with_args(&skill, Some("src/foo.rs --strict")).unwrap();

        assert!(rendered.contains("Review src/foo.rs --strict carefully."));
    }

    #[test]
    fn loaded_skill_context_with_args_uses_context_without_args() {
        let skill = skill_with_content("Review the code carefully.");

        let rendered = loaded_skill_context_with_args(&skill, None).unwrap();

        assert!(rendered.contains("# Loaded Skill: test-skill (skill)"));
        assert!(rendered.contains("## Content\n\nReview the code carefully."));
    }

    #[test]
    fn loaded_skill_context_shows_resolved_paths_for_supporting_files() {
        let skill_dir = std::env::temp_dir().join("goose-test-skill");
        let script_path = skill_dir.join("scripts").join("my-tool.exe");
        let mut skill = skill_with_content("Run scripts/my-tool.exe.");
        skill.path = skill_dir.to_string_lossy().into_owned();
        skill.supporting_files = vec![script_path.to_string_lossy().into_owned()];

        let rendered = loaded_skill_context_with_args(&skill, None).unwrap();
        let resolved_path = script_path.to_string_lossy().replace('\\', "/");

        assert!(rendered.contains("Relative paths in this skill resolve from the skill directory"));
        assert!(rendered.contains("scripts/my-tool.exe"));
        assert!(rendered.contains(&resolved_path));
        assert!(rendered.contains("load_skill(name: \"test-skill/scripts/my-tool.exe\")"));
    }

    #[test]
    fn resolve_docs_root_placeholder_substitutes_builtin_goose_doc_guide_root() {
        let skill = builtin_goose_doc_guide_skill();

        let rendered =
            resolve_docs_root_placeholder(&skill, &skill.content, "/tmp/goose docs/root");

        assert_eq!(rendered, "Docs root: /tmp/goose docs/root.");
    }

    #[test]
    fn resolve_docs_root_placeholder_ignores_non_builtin_goose_doc_guide_skills() {
        let mut skill = builtin_goose_doc_guide_skill();
        skill.source_type = SourceType::Skill;

        let rendered = resolve_docs_root_placeholder(&skill, &skill.content, "/tmp/goose-docs");

        assert_eq!(rendered, skill.content);
    }

    #[test]
    fn project_plugin_skill_precedes_global_skill_with_same_name() {
        let project = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let plugin_root = project.path().join(".agents/plugins/project-plugin");
        write_test_skill(
            &plugin_root.join("skills/collision"),
            "collision",
            "project plugin body",
        );
        write_test_skill(
            &path_root.path().join("config/skills/collision"),
            "collision",
            "global body",
        );

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        config
            .set_param(
                "plugins",
                HashMap::from([(
                    plugin_root.to_string_lossy().into_owned(),
                    HashMap::from([("enabled", true)]),
                )]),
            )
            .unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);

        let skill = discover_skills_with_config(Some(project.path()), &config)
            .into_iter()
            .find(|skill| skill.name == "collision")
            .unwrap();

        assert_eq!(skill.content.trim(), "project plugin body");
        assert!(!skill.global);
        assert!(!skill.writable);
        assert!(Path::new(&skill.path).starts_with(&plugin_root));
    }

    #[test]
    fn disabled_skills_are_hidden_from_discovery_but_listed() {
        let project = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        write_test_skill(&path_root.path().join("config/skills/mine"), "mine", "body");
        write_test_skill(
            &path_root.path().join("config/skills/other"),
            "other",
            "body",
        );

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        config
            .set_param("GOOSE_DISABLED_SKILLS", vec!["mine", "goose-doc-guide"])
            .unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("GOOSE_DISABLED_SKILLS", None),
            ("GOOSE_SKILL_LISTING", None),
        ]);

        let names = |skills: Vec<SourceEntry>| -> HashSet<String> {
            skills.into_iter().map(|skill| skill.name).collect()
        };
        let active = names(discover_skills_with_config(Some(project.path()), &config));
        assert!(active.contains("other"));
        assert!(active.contains("web-search"));
        assert!(!active.contains("mine"));
        assert!(!active.contains("goose-doc-guide"));

        let all = names(discover_all_skills_with_config(
            Some(project.path()),
            &config,
        ));
        assert!(all.contains("mine"));
        assert!(all.contains("goose-doc-guide"));
    }

    #[test]
    fn user_plugin_skill_remains_writable_when_project_root_is_path_root() {
        let path_root = tempfile::tempdir().unwrap();
        let plugin_root = path_root.path().join(".agents/plugins/user-plugin");
        write_test_skill(
            &plugin_root.join("skills/user-owned"),
            "user-owned",
            "user body",
        );

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        config
            .set_param(
                "plugins",
                HashMap::from([(
                    plugin_root.to_string_lossy().into_owned(),
                    HashMap::from([("enabled", true)]),
                )]),
            )
            .unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);

        let skill = discover_skills_with_config(Some(path_root.path()), &config)
            .into_iter()
            .find(|skill| skill.name == "user-owned")
            .unwrap();

        assert!(skill.global);
        assert!(skill.writable);
    }

    #[test]
    fn exclusive_project_plugin_manifest_omits_default_skill_root() {
        let project = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let plugin_root = project.path().join(".agents/plugins/project-plugin");
        write_test_skill(
            &plugin_root.join("skills/excluded"),
            "excluded",
            "excluded body",
        );
        write_test_skill(
            &plugin_root.join("custom-skills/included"),
            "included",
            "included body",
        );
        std::fs::write(
            plugin_root.join("plugin.json"),
            r#"{"name":"project-plugin","skills":{"exclusive":true,"paths":["./custom-skills"]}}"#,
        )
        .unwrap();

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        config
            .set_param(
                "plugins",
                HashMap::from([(
                    plugin_root.to_string_lossy().into_owned(),
                    HashMap::from([("enabled", true)]),
                )]),
            )
            .unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);

        let skills = discover_skills_with_config(Some(project.path()), &config);

        assert!(skills.iter().any(|skill| skill.name == "included"));
        assert!(!skills.iter().any(|skill| skill.name == "excluded"));
    }

    #[test]
    fn project_plugin_skill_is_rejected_by_source_crud_before_discovery() {
        let project = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let plugin_root = project.path().join(".agents/plugins/project-plugin");
        let skill_dir = plugin_root.join(".agents/skills/plugin-owned");
        write_test_skill(&skill_dir, "plugin-owned", "plugin body");
        std::fs::write(
            plugin_root.join("plugin.json"),
            r#"{"name":"project-plugin","skills":{"paths":["./.agents/skills"]}}"#,
        )
        .unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);

        assert_path_rejected_by_source_crud(
            skill_dir.to_str().unwrap(),
            "plugin-owned",
            &skill_dir,
        );
    }

    #[cfg(unix)]
    #[test]
    fn configured_disabled_project_plugin_canonical_path_is_rejected() {
        let project = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let external_root = tempfile::tempdir().unwrap();
        let plugin_link = project.path().join(".agents/plugins/project-plugin");
        let skill_dir = external_root.path().join(".agents/skills/plugin-owned");
        write_test_skill(&skill_dir, "plugin-owned", "plugin body");
        std::fs::write(
            external_root.path().join("plugin.json"),
            r#"{"name":"project-plugin","skills":{"paths":["./.agents/skills"]}}"#,
        )
        .unwrap();
        std::fs::create_dir_all(plugin_link.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(external_root.path(), &plugin_link).unwrap();
        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        config
            .set_param(
                "plugins",
                HashMap::from([(
                    plugin_link.to_string_lossy().into_owned(),
                    HashMap::from([("enabled", false)]),
                )]),
            )
            .unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);

        let err = resolve_discoverable_skill_dir_with_config(skill_dir.to_str().unwrap(), &config)
            .unwrap_err();

        assert!(format!("{err:?}").contains("not found"));
        assert!(skill_dir.join("SKILL.md").is_file());
    }

    #[cfg(unix)]
    #[test]
    fn settings_disabled_project_plugin_canonical_path_is_rejected() {
        let project = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let external_root = tempfile::tempdir().unwrap();
        let plugin_link = project.path().join(".agents/plugins/project-plugin");
        let skill_dir = external_root.path().join(".agents/skills/plugin-owned");
        write_test_skill(&skill_dir, "plugin-owned", "plugin body");
        std::fs::write(
            external_root.path().join("plugin.json"),
            r#"{"name":"project-plugin","skills":{"paths":["./.agents/skills"]}}"#,
        )
        .unwrap();
        std::fs::create_dir_all(plugin_link.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(external_root.path(), &plugin_link).unwrap();
        let settings_dir = project.path().join(".config/goose");
        std::fs::create_dir_all(&settings_dir).unwrap();
        std::fs::write(
            settings_dir.join("settings.json"),
            r#"{"disabledPlugins":["project-plugin"]}"#,
        )
        .unwrap();
        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);

        let plugins = crate::plugins::discovery::discover_enabled_plugins_with_config(
            Some(project.path()),
            &config,
        );
        assert!(plugins.iter().all(|plugin| plugin.name != "project-plugin"));

        let err = resolve_discoverable_skill_dir_with_config(skill_dir.to_str().unwrap(), &config)
            .unwrap_err();
        assert!(format!("{err:?}").contains("not found"));
        assert!(skill_dir.join("SKILL.md").is_file());
    }

    #[test]
    fn nested_project_plugin_skill_is_listed_read_only_and_rejected_by_source_crud() {
        let project = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let plugin_root = project.path().join(".agents/plugins/project-plugin");
        let skill_dir = plugin_root.join(".agents/skills/plugin-owned");
        write_test_skill(&skill_dir, "plugin-owned", "plugin body");
        std::fs::write(
            plugin_root.join("plugin.json"),
            r#"{"name":"project-plugin","skills":{"paths":["./.agents/skills"]}}"#,
        )
        .unwrap();

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        config
            .set_param(
                "plugins",
                HashMap::from([(
                    plugin_root.to_string_lossy().into_owned(),
                    HashMap::from([("enabled", true)]),
                )]),
            )
            .unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);

        let skill = discover_skills_with_config(Some(project.path()), &config)
            .into_iter()
            .find(|skill| skill.name == "plugin-owned")
            .unwrap();
        assert_read_only_and_rejected_by_source_crud(&skill, &skill_dir);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_project_plugin_skill_is_rejected_by_source_crud() {
        let project = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let external_root = tempfile::tempdir().unwrap();
        let plugin_link = project.path().join(".agents/plugins/project-plugin");
        let skill_dir = external_root.path().join(".agents/skills/plugin-owned");
        write_test_skill(&skill_dir, "plugin-owned", "plugin body");
        std::fs::write(
            external_root.path().join("plugin.json"),
            r#"{"name":"project-plugin","skills":{"paths":["./.agents/skills"]}}"#,
        )
        .unwrap();
        std::fs::create_dir_all(plugin_link.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(external_root.path(), &plugin_link).unwrap();

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        config
            .set_param(
                "plugins",
                HashMap::from([(
                    plugin_link.to_string_lossy().into_owned(),
                    HashMap::from([("enabled", true)]),
                )]),
            )
            .unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);

        let skill = discover_skills_with_config(Some(project.path()), &config)
            .into_iter()
            .find(|skill| skill.name == "plugin-owned")
            .unwrap();

        assert!(Path::new(&skill.path).starts_with(&plugin_link));
        assert_read_only_and_rejected_by_source_crud(&skill, &skill_dir);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_user_plugin_skill_remains_writable_for_source_crud() {
        let path_root = tempfile::tempdir().unwrap();
        let external_parent = tempfile::tempdir().unwrap();
        let external_root = external_parent.path().join(".agents/plugins/user-plugin");
        let skill_dir = external_root.join(".agents/skills/user-owned");
        write_test_skill(&skill_dir, "user-owned", "user body");
        std::fs::write(
            external_root.join("plugin.json"),
            r#"{"name":"user-plugin","skills":{"paths":["./.agents/skills"]}}"#,
        )
        .unwrap();

        let project = tempfile::tempdir().unwrap();
        let project_path_root = tempfile::tempdir().unwrap();
        let project_plugin_link = project.path().join(".agents/plugins/user-plugin");
        std::fs::create_dir_all(project_plugin_link.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(&external_root, &project_plugin_link).unwrap();
        let project_config = Config::new(
            project_path_root.path().join("test-config.yaml"),
            "skills-test",
        )
        .unwrap();
        project_config
            .set_param(
                "plugins",
                HashMap::from([(
                    project_plugin_link.to_string_lossy().into_owned(),
                    HashMap::from([("enabled", true)]),
                )]),
            )
            .unwrap();
        {
            let _guard = env_lock::lock_env([
                ("GOOSE_PATH_ROOT", project_path_root.path().to_str()),
                ("PLUGINS", None),
            ]);
            let project_skill = discover_skills_with_config(Some(project.path()), &project_config)
                .into_iter()
                .find(|skill| skill.name == "user-owned")
                .unwrap();
            assert!(!project_skill.writable);
        }

        let plugin_link = path_root.path().join(".agents/plugins/user-plugin");
        std::fs::create_dir_all(plugin_link.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(&external_root, &plugin_link).unwrap();

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        config
            .set_param(
                "plugins",
                HashMap::from([(
                    plugin_link.to_string_lossy().into_owned(),
                    HashMap::from([("enabled", true)]),
                )]),
            )
            .unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);

        let skill = discover_skills_with_config(None, &config)
            .into_iter()
            .find(|skill| skill.name == "user-owned")
            .unwrap();
        assert!(skill.global);
        assert!(skill.writable);
        assert!(Path::new(&skill.path).starts_with(&plugin_link));
        assert!(resolve_discoverable_skill_dir_with_config(&skill.path, &project_config).is_ok());

        let updated = crate::sources::update_source_with_roots(
            SourceType::Skill,
            &skill.path,
            "user-owned",
            "updated",
            "updated body",
            crate::sources::UpdateSourceOptions {
                properties: Some(HashMap::new()),
                additional_roots: &[],
            },
        )
        .unwrap();
        assert_eq!(updated.content, "updated body");

        let (exported, _) = crate::sources::export_source(SourceType::Skill, &skill.path).unwrap();
        assert!(exported.contains("updated body"));

        crate::sources::delete_source(SourceType::Skill, &skill.path).unwrap();
        assert!(!skill_dir.exists());
    }

    #[test]
    fn ordinary_project_skill_under_plugin_manifest_remains_writable() {
        let project = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let skill_dir = project.path().join(".agents/skills/project-owned");
        write_test_skill(&skill_dir, "project-owned", "project body");
        std::fs::write(
            project.path().join("plugin.json"),
            r#"{"name":"ordinary-project","skills":{"paths":["./.agents/skills"]}}"#,
        )
        .unwrap();
        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);

        let skill = discover_skills_with_config(Some(project.path()), &config)
            .into_iter()
            .find(|skill| skill.name == "project-owned")
            .unwrap();
        assert!(skill.writable);

        let updated = crate::sources::update_source_with_roots(
            SourceType::Skill,
            &skill.path,
            "project-owned",
            "updated",
            "updated body",
            crate::sources::UpdateSourceOptions {
                properties: Some(HashMap::new()),
                additional_roots: &[],
            },
        )
        .unwrap();
        assert_eq!(updated.content, "updated body");

        let (exported, _) = crate::sources::export_source(SourceType::Skill, &skill.path).unwrap();
        assert!(exported.contains("updated body"));

        crate::sources::delete_source(SourceType::Skill, &skill.path).unwrap();
        assert!(!skill_dir.exists());
    }

    #[test]
    fn claude_profile_reads_claude_locations_one_level_deep() {
        let home = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let personal = home.path().join(".claude/skills");
        write_test_skill(
            &personal.join("personal-skill"),
            "personal-skill",
            "personal",
        );
        write_test_skill(&personal.join("shared"), "shared", "personal shared");
        write_test_skill(
            &personal.join("personal-skill/bundles/nested-skill"),
            "nested-skill",
            "nested",
        );
        write_test_skill(
            &personal.join("_Archiv/archived-skill"),
            "archived-skill",
            "archived",
        );
        write_test_skill(
            &personal.join("synced/account_pair/synced-skill"),
            "synced-skill",
            "synced",
        );
        write_test_skill(
            &home.path().join(".agents/skills/agents-skill"),
            "agents-skill",
            "agents",
        );
        write_test_skill(
            &project.path().join(".claude/skills/project-skill"),
            "project-skill",
            "project",
        );
        write_test_skill(
            &project.path().join(".claude/skills/shared"),
            "shared",
            "project shared",
        );
        write_test_skill(
            &project.path().join(".agents/skills/project-agents-skill"),
            "project-agents-skill",
            "agents",
        );
        write_test_skill(
            &project.path().join(".goose/skills/project-goose-skill"),
            "project-goose-skill",
            "goose",
        );

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);
        let dirs = skill_dirs_with_config(true, Some(home.path()), Some(project.path()), &config);
        let skills = discover_skills_in_dirs(dirs);

        let names: Vec<&str> = skills.iter().map(|skill| skill.name.as_str()).collect();
        assert!(names.contains(&"personal-skill"), "{names:?}");
        assert!(names.contains(&"project-skill"), "{names:?}");
        for absent in [
            "nested-skill",
            "archived-skill",
            "synced-skill",
            "agents-skill",
            "project-agents-skill",
            "project-goose-skill",
        ] {
            assert!(!names.contains(&absent), "{absent} listed: {names:?}");
        }

        let shared = skills.iter().find(|skill| skill.name == "shared").unwrap();
        assert_eq!(shared.content.trim(), "personal shared");
        assert!(
            Path::new(&shared.path).starts_with(personal.join("shared").canonicalize().unwrap())
        );
    }

    #[test]
    fn goose_profile_keeps_recursive_agent_roots() {
        let home = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        write_test_skill(
            &home.path().join(".agents/skills/agents-skill"),
            "agents-skill",
            "agents",
        );
        write_test_skill(
            &home.path().join(".agents/skills/_Archiv/archived-skill"),
            "archived-skill",
            "archived",
        );
        write_test_skill(
            &project.path().join(".agents/skills/project-skill"),
            "project-skill",
            "project",
        );

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);
        let dirs = skill_dirs_with_config(false, Some(home.path()), Some(project.path()), &config);

        assert_eq!(
            dirs.first().map(|dir| dir.path.clone()),
            Some(project.path().join(".agents/skills"))
        );
        assert!(dirs
            .iter()
            .any(|dir| dir.path == home.path().join(".agents/skills") && dir.recursive));

        let skills = discover_skills_in_dirs(dirs);
        let names: Vec<&str> = skills.iter().map(|skill| skill.name.as_str()).collect();
        assert!(names.contains(&"agents-skill"), "{names:?}");
        assert!(names.contains(&"archived-skill"), "{names:?}");
        assert!(names.contains(&"project-skill"), "{names:?}");
    }

    #[test]
    fn claude_profile_names_plugin_skills_by_plugin() {
        let home = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let plugins = path_root.path().join("config/skills/plugins");
        write_test_skill(
            &plugins.join("expo/eas-app-stores"),
            "eas-app-stores",
            "expo body",
        );
        write_test_skill(
            &plugins.join("expo/expo-router"),
            "expo:expo-router",
            "already prefixed",
        );
        write_test_skill(&plugins.join("pdf-viewer/annotate"), "annotate", "pdf body");
        write_test_skill(&plugins.join("anthropic-skills/pdf"), "pdf", "synced body");
        write_test_skill(
            &path_root.path().join("config/skills/befehle/annotate"),
            "annotate",
            "command body",
        );

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);
        let dirs = skill_dirs_with_config(true, Some(home.path()), None, &config);
        let skills = discover_skills_in_dirs(dirs);

        let names: Vec<&str> = skills.iter().map(|skill| skill.name.as_str()).collect();
        assert!(names.contains(&"expo:eas-app-stores"), "{names:?}");
        assert!(names.contains(&"expo:expo-router"), "{names:?}");
        assert_eq!(
            names
                .iter()
                .filter(|name| **name == "expo:expo-router")
                .count(),
            1,
            "{names:?}"
        );
        assert!(names.contains(&"pdf-viewer:annotate"), "{names:?}");
        assert!(names.contains(&"anthropic-skills:pdf"), "{names:?}");
        assert!(names.contains(&"annotate"), "{names:?}");
        assert!(!names.contains(&"eas-app-stores"), "{names:?}");
    }

    #[test]
    fn claude_profile_names_a_skill_after_its_folder_whatever_its_frontmatter() {
        let home = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let personal = home.path().join(".claude/skills");
        write_test_skill(&personal.join("folder-name"), "frontmatter-name", "renamed");
        std::fs::create_dir_all(personal.join("no-frontmatter")).unwrap();
        std::fs::write(personal.join("no-frontmatter/SKILL.md"), "just text").unwrap();
        std::fs::create_dir_all(personal.join("broken-frontmatter")).unwrap();
        std::fs::write(
            personal.join("broken-frontmatter/SKILL.md"),
            "---\ndescription: one: two: [\n---\nbody",
        )
        .unwrap();

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);
        let dirs = skill_dirs_with_config(true, Some(home.path()), None, &config);
        let skills = discover_skills_in_dirs(dirs);

        let names: Vec<&str> = skills.iter().map(|skill| skill.name.as_str()).collect();
        assert!(names.contains(&"folder-name"), "{names:?}");
        assert!(!names.contains(&"frontmatter-name"), "{names:?}");
        assert!(names.contains(&"no-frontmatter"), "{names:?}");
        assert!(names.contains(&"broken-frontmatter"), "{names:?}");
    }

    #[test]
    fn a_bare_name_reaches_the_one_namespaced_skill_answering_to_it() {
        let skill = |name: &str| SourceEntry {
            source_type: SourceType::Skill,
            name: name.to_string(),
            description: String::new(),
            content: String::new(),
            path: String::new(),
            global: true,
            writable: false,
            supporting_files: Vec::new(),
            properties: HashMap::new(),
        };
        let skills = vec![
            skill("inventar"),
            skill("anthropic-skills:inventar"),
            skill("anthropic-skills:pdf"),
            skill("design:ux-copy"),
            skill("expo:ux-copy"),
        ];
        let found = |name: &str| find_skill(&skills, name).map(|skill| skill.name.as_str());

        assert_eq!(found("inventar"), Some("inventar"));
        assert_eq!(
            found("anthropic-skills:inventar"),
            Some("anthropic-skills:inventar")
        );
        assert_eq!(found("pdf"), Some("anthropic-skills:pdf"));
        assert_eq!(found("ux-copy"), None);
        assert_eq!(found("other:pdf"), None);
    }

    #[test]
    fn claude_profile_reads_every_parent_project_folder_nearest_first() {
        let home = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let top = home.path().join("work");
        write_test_skill(&top.join(".claude/skills/top-skill"), "top-skill", "top");
        write_test_skill(&top.join(".claude/skills/shared"), "shared", "top shared");
        write_test_skill(
            &top.join("sub/.claude/skills/sub-skill"),
            "sub-skill",
            "sub",
        );
        write_test_skill(
            &top.join("sub/.claude/skills/shared"),
            "shared",
            "sub shared",
        );
        write_test_skill(
            &top.join("sub/deep/.claude/skills/deep-skill"),
            "deep-skill",
            "deep",
        );
        let working_dir = top.join("sub/deep/nested");
        std::fs::create_dir_all(&working_dir).unwrap();

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);
        let dirs = skill_dirs_with_config(
            true,
            Some(home.path()),
            Some(working_dir.as_path()),
            &config,
        );
        let skills = discover_skills_in_dirs(dirs);

        let index = |name: &str| {
            skills
                .iter()
                .position(|skill| skill.name == name)
                .unwrap_or_else(|| {
                    panic!(
                        "{name} missing: {:?}",
                        skills.iter().map(|s| &s.name).collect::<Vec<_>>()
                    )
                })
        };
        assert!(index("deep-skill") < index("sub-skill"));
        assert!(index("sub-skill") < index("top-skill"));
        let shared = skills.iter().find(|skill| skill.name == "shared").unwrap();
        assert_eq!(shared.content.trim(), "sub shared");
    }

    #[test]
    fn claude_profile_stops_project_folders_at_the_git_root() {
        let home = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let work = home.path().join("work");
        write_test_skill(
            &work.join(".claude/skills/above-repo-skill"),
            "above-repo-skill",
            "above",
        );
        let repo = work.join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        write_test_skill(
            &repo.join(".claude/skills/repo-skill"),
            "repo-skill",
            "repo",
        );
        let working_dir = repo.join("crate/deep");
        std::fs::create_dir_all(&working_dir).unwrap();

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);
        let dirs = skill_dirs_with_config(
            true,
            Some(home.path()),
            Some(working_dir.as_path()),
            &config,
        );
        let names: Vec<String> = discover_skills_in_dirs(dirs)
            .into_iter()
            .map(|skill| skill.name)
            .collect();
        assert!(names.contains(&"repo-skill".to_string()), "{names:?}");
        assert!(
            !names.contains(&"above-repo-skill".to_string()),
            "{names:?}"
        );
    }

    #[test]
    fn claude_profile_skips_reserved_synced_folders_but_reads_dot_named_skills() {
        let home = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let path_root = tempfile::tempdir().unwrap();
        let personal = home.path().join(".claude/skills");
        write_test_skill(&personal.join("Synced"), "reserved-probe", "reserved");
        write_test_skill(&personal.join(".hidden-probe"), "hidden-probe", "hidden");
        write_test_skill(
            &project.path().join(".claude/skills/synced"),
            "project-reserved-probe",
            "project reserved",
        );
        write_test_skill(
            &project.path().join(".claude/skills/.project-hidden"),
            "project-hidden",
            "project hidden",
        );

        let config = Config::new(path_root.path().join("test-config.yaml"), "skills-test").unwrap();
        let _guard = env_lock::lock_env([
            ("GOOSE_PATH_ROOT", path_root.path().to_str()),
            ("PLUGINS", None),
        ]);
        let dirs = skill_dirs_with_config(true, Some(home.path()), Some(project.path()), &config);
        let names: Vec<String> = discover_skills_in_dirs(dirs)
            .into_iter()
            .map(|skill| skill.name)
            .collect();
        assert!(names.contains(&".hidden-probe".to_string()), "{names:?}");
        assert!(names.contains(&".project-hidden".to_string()), "{names:?}");
        assert!(!names.contains(&"reserved-probe".to_string()), "{names:?}");
        assert!(
            !names.contains(&"project-reserved-probe".to_string()),
            "{names:?}"
        );
    }

    #[test]
    fn claude_profile_skips_a_skill_file_an_earlier_root_already_supplied() {
        let personal = tempfile::tempdir().unwrap();
        write_test_skill(&personal.path().join("shared-file"), "shared-file", "body");
        std::fs::create_dir_all(personal.path().join("notes")).unwrap();
        let roots = vec![
            SkillRoot {
                path: personal.path().to_path_buf(),
                recursive: false,
                naming: SkillNaming::Folder,
                global: true,
                writable: true,
                preserve_path: false,
            },
            SkillRoot {
                path: personal.path().join("notes").join(".."),
                recursive: false,
                naming: SkillNaming::PluginFolder("design".to_string()),
                global: true,
                writable: true,
                preserve_path: false,
            },
        ];

        let names: Vec<String> = goose_agent::skills::discover_skills(roots)
            .into_iter()
            .map(|skill| skill.name)
            .collect();
        assert!(names.contains(&"shared-file".to_string()), "{names:?}");
        assert!(
            !names.contains(&"design:shared-file".to_string()),
            "{names:?}"
        );
    }
}
