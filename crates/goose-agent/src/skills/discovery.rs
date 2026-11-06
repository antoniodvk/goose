//! Native filesystem skill discovery under caller-supplied roots.

use goose_sdk_types::custom_requests::{SourceEntry, SourceType};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tracing::warn;

/// How the entries of a root are named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillNaming {
    /// The `name` in the frontmatter.
    Frontmatter,
    /// The skill folder's name, which is what Claude Code goes by. Such a skill needs
    /// no `name`, and it stays listed when its frontmatter cannot be read.
    Folder,
    /// `<plugin>:<folder>`, Claude Code's name for a plugin skill.
    PluginFolder(String),
}

/// A host-selected search root; earlier roots take precedence.
#[derive(Debug, Clone)]
pub struct SkillRoot {
    /// Directory searched for `SKILL.md`. Returned entry paths identify
    /// the containing skill directory, not the Markdown file.
    /// Relative roots are resolved against the process working directory.
    pub path: PathBuf,
    /// Descend into subdirectories looking for `SKILL.md`. With `false` only a
    /// `SKILL.md` directly inside a child directory of the root is found, which is
    /// how Claude Code treats its `skills` folders.
    pub recursive: bool,
    pub naming: SkillNaming,
    /// Scope metadata copied to each entry, not inferred from its location.
    pub global: bool,
    /// Writability metadata copied to each entry. Discovery does not check
    /// permissions or enforce whether a host may modify the skill.
    pub writable: bool,
    /// Retain lexical skill-directory and supporting-file paths (e.g. plugin links)
    /// instead of canonicalizing the skill directory. Relative roots then produce
    /// relative result paths; supply absolute roots when absolute paths are needed.
    /// This does not change symlink traversal or confine reads to the root.
    pub preserve_path: bool,
}

#[derive(Debug, Default, Deserialize)]
struct Frontmatter {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: String,
    #[serde(default)]
    metadata: HashMap<String, Value>,
}

fn parse_frontmatter<T: for<'de> Deserialize<'de>>(
    content: &str,
) -> Result<Option<(T, String)>, serde_yaml::Error> {
    let parts: Vec<&str> = content.split("---").collect();
    if parts.len() < 3 {
        return Ok(None);
    }

    let yaml_content = parts[1].trim();
    let metadata: T = serde_yaml::from_str(yaml_content)?;

    let body = parts[2..].join("---").trim().to_string();
    Ok(Some((metadata, body)))
}

fn parse_skill_content(
    content: &str,
    path: &Path,
    global: bool,
    writable: bool,
    folder_name: Option<String>,
) -> Option<SourceEntry> {
    let parsed = parse_frontmatter(content).unwrap_or_else(|e| {
        warn!("Failed to parse skill frontmatter: {}", e);
        None
    });
    let (metadata, body): (Frontmatter, String) = match parsed {
        Some(parsed) => parsed,
        None if folder_name.is_some() => (Frontmatter::default(), content.to_string()),
        None => return None,
    };

    let name = match folder_name.or(metadata.name.filter(|n| !n.is_empty())) {
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

fn should_skip_dir(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(".git") | Some(".hg") | Some(".svn")
    )
}

fn walk_files_recursively<F, G>(
    dir: &Path,
    visited_dirs: &mut HashSet<PathBuf>,
    should_descend: &mut G,
    visit_file: &mut F,
) where
    F: FnMut(&Path),
    G: FnMut(&Path) -> bool,
{
    let canonical_dir = match std::fs::canonicalize(dir) {
        Ok(path) => path,
        Err(_) => return,
    };

    if !visited_dirs.insert(canonical_dir) {
        return;
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if should_descend(&path) {
                walk_files_recursively(&path, visited_dirs, should_descend, visit_file);
            }
        } else if path.is_file() {
            visit_file(&path);
        }
    }
}

/// Claude Code keeps this folder name for the sync-owned root and never treats it as
/// a skill, comparing it in its normalized form, which ignores case.
const SYNCED_DIR_NAME: &str = "synced";

fn skill_files_in_root(dir: &Path, recursive: bool) -> Vec<PathBuf> {
    let mut skill_files = Vec::new();
    if recursive {
        let mut visited_dirs = HashSet::new();
        walk_files_recursively(
            dir,
            &mut visited_dirs,
            &mut |path| !should_skip_dir(path),
            &mut |path| {
                if path.file_name().and_then(|name| name.to_str()) == Some("SKILL.md") {
                    skill_files.push(path.to_path_buf());
                }
            },
        );
    } else if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.eq_ignore_ascii_case(SYNCED_DIR_NAME))
            {
                continue;
            }
            let skill_file = entry.path().join("SKILL.md");
            if skill_file.is_file() {
                skill_files.push(skill_file);
            }
        }
    }
    skill_files
}

fn scan_skills_from_dir(
    root: &SkillRoot,
    seen_names: &mut HashSet<String>,
    seen_files: &mut HashSet<PathBuf>,
) -> Vec<SourceEntry> {
    let skill_files = skill_files_in_root(&root.path, root.recursive);

    let mut sources = Vec::new();
    for skill_file in skill_files {
        let Some(skill_dir) = skill_file.parent() else {
            continue;
        };
        let Ok(file_identity) = skill_file.canonicalize() else {
            continue;
        };
        if seen_files.contains(&file_identity) {
            continue;
        }
        let registered_dir = if root.preserve_path {
            skill_dir.to_path_buf()
        } else {
            let Ok(canonical_dir) = skill_dir.canonicalize() else {
                continue;
            };
            canonical_dir
        };
        let content = match std::fs::read_to_string(&skill_file) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to read skill file {}: {}", skill_file.display(), e);
                continue;
            }
        };

        if let Some(mut source) = parse_skill_content(
            &content,
            &registered_dir,
            root.global,
            root.writable,
            folder_name(&root.naming, skill_dir),
        ) {
            if !seen_names.contains(&source.name) {
                let mut files = Vec::new();
                let mut visited_support_dirs = HashSet::new();
                walk_files_recursively(
                    skill_dir,
                    &mut visited_support_dirs,
                    &mut |path| !should_skip_dir(path) && !path.join("SKILL.md").is_file(),
                    &mut |path| {
                        if path.file_name().and_then(|n| n.to_str()) != Some("SKILL.md") {
                            if let Ok(relative) = path.strip_prefix(skill_dir) {
                                files.push(
                                    registered_dir.join(relative).to_string_lossy().into_owned(),
                                );
                            }
                        }
                    },
                );
                source.supporting_files = files;

                seen_names.insert(source.name.clone());
                seen_files.insert(file_identity);
                sources.push(source);
            }
        }
    }
    sources
}

/// The name a root's naming takes from the skill folder; `None` leaves it to the frontmatter.
fn folder_name(naming: &SkillNaming, skill_dir: &Path) -> Option<String> {
    let folder = skill_dir.file_name()?.to_str()?;
    match naming {
        SkillNaming::Frontmatter => None,
        SkillNaming::Folder => Some(folder.to_string()),
        SkillNaming::PluginFolder(plugin) => Some(format!("{plugin}:{folder}")),
    }
}

/// Discovers skills from explicit roots, without consulting host configuration. Each
/// root says whether it is searched recursively or only one level deep.
///
/// Roots are visited in input order. A skill file an earlier root already supplied
/// (the same file after resolving links) is skipped, as Claude Code does; after that
/// the first successfully parsed entry with a given name wins, names compared exactly
/// and case-sensitively. Each root says how its entries are named.
/// Traversal within each root is unsorted, so result order, duplicate-name winners
/// within a root, and supporting-file order are unspecified.
///
/// Discovery is best-effort: missing or unreadable directories and invalid or
/// unreadable skills are skipped. Some failures emit tracing warnings; no errors
/// are returned. An empty result does not distinguish no skills from a failed scan.
///
/// Traversal follows file and directory symlinks, including targets outside a
/// root. Canonical directory identities prevent cycles. `preserve_path` only
/// affects returned path spelling; discovery is not a confined file-open API.
/// The caller must enforce access policy when opening returned paths.
pub fn discover_skills(roots: impl IntoIterator<Item = SkillRoot>) -> Vec<SourceEntry> {
    let mut sources: Vec<SourceEntry> = Vec::new();
    let mut seen_names = HashSet::new();
    let mut seen_files = HashSet::new();

    for root in roots {
        for source in scan_skills_from_dir(&root, &mut seen_names, &mut seen_files) {
            sources.push(source);
        }
    }

    sources
}
