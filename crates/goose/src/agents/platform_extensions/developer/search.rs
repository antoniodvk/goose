//! Read-only search tools like Claude Code's Grep and Glob: content search with a regular
//! expression and file name search with glob patterns. Both walk with the same rules as tree
//! (.gitignore and .ignore without requiring git, hidden entries included, never .git) and never
//! follow symbolic links, so a link into a profile or secrets folder is not searched.
//!
//! Read rules from Claude Code's `permissions.deny` also apply here: the files named in
//! `GOOSE_READ_DENY_FILES` (JSON list of settings.json paths) are read on every call, and files
//! matching a `Read(...)` rule are left out of the results. If a listed file cannot be read, the
//! search refuses to run (fail closed).

use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use ignore::overrides::OverrideBuilder;
use ignore::types::TypesBuilder;
use ignore::{DirEntry, WalkBuilder};
use regex::{Regex, RegexBuilder};
use rmcp::model::{CallToolResult, ContentBlock};
use schemars::JsonSchema;
use serde::Deserialize;

use super::edit::resolve_path;
use super::encoding::Bom;
use super::shell::OUTPUT_LIMIT_BYTES;

pub const READ_DENY_FILES_ENV: &str = "GOOSE_READ_DENY_FILES";
const DEFAULT_HEAD_LIMIT: usize = 250;
const GLOB_LIMIT: usize = 100;
const MAX_LINE_CHARS: usize = 500;
const MAX_SEARCHED_FILE_BYTES: u64 = 20 * 1024 * 1024;
const BINARY_PROBE_BYTES: usize = 8192;

#[derive(Debug, Clone, Copy, Default, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    /// Matching lines as `path:line:text`.
    Content,
    /// Only the paths of files with a match, most recently modified first (default).
    #[default]
    FilesWithMatches,
    /// Number of matching lines per file as `path:count`.
    Count,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GrepParams {
    /// Regular expression to search for (Rust regex syntax, like ripgrep). Escape literal braces, e.g. `interface\{\}`.
    pub pattern: String,
    /// File or directory to search in. Defaults to the current working directory.
    #[serde(default)]
    pub path: Option<String>,
    /// Only search files matching this glob, e.g. `*.mjs` or `**/*.{ts,tsx}`.
    #[serde(default)]
    pub glob: Option<String>,
    /// Only search files of this type, e.g. js, ts, rust, py, md (ripgrep type names).
    #[serde(default, rename = "type")]
    pub file_type: Option<String>,
    /// content, files_with_matches (default) or count.
    #[serde(default)]
    pub output_mode: OutputMode,
    /// Lines of context after each match (content mode).
    #[serde(default, rename = "-A")]
    pub after: Option<usize>,
    /// Lines of context before each match (content mode).
    #[serde(default, rename = "-B")]
    pub before: Option<usize>,
    /// Lines of context before and after each match (content mode).
    #[serde(default, rename = "-C")]
    pub context_short: Option<usize>,
    /// Same as -C.
    #[serde(default)]
    pub context: Option<usize>,
    /// Case-insensitive search.
    #[serde(default, rename = "-i")]
    pub case_insensitive: Option<bool>,
    /// Show line numbers (content mode, default true).
    #[serde(default, rename = "-n")]
    pub line_numbers: Option<bool>,
    /// Let the pattern span lines; `.` also matches a line break.
    #[serde(default)]
    pub multiline: Option<bool>,
    /// Return at most this many lines or entries (default 250, 0 for no limit).
    #[serde(default)]
    pub head_limit: Option<usize>,
    /// Skip this many lines or entries first, to continue a limited result.
    #[serde(default)]
    pub offset: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GlobParams {
    /// Glob pattern for file names, e.g. `**/*.ts` or `src/**/*.mjs`.
    pub pattern: String,
    /// Directory to search in. Defaults to the current working directory.
    #[serde(default)]
    pub path: Option<String>,
}

pub struct SearchTools;

impl Default for SearchTools {
    fn default() -> Self {
        Self::new()
    }
}

fn error(text: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(text.into())])
}

fn success(text: impl Into<String>) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(text.into())])
}

/// Patterns of `Read(...)` deny rules, converted like the server's matchDeny
/// (server/code-chats.mjs): `//` at the start means an absolute path, `**` is the same as `*`,
/// a pattern without drive letter matches at the end of any path, case-insensitive.
struct DenyList(Vec<Regex>);

impl DenyList {
    fn load() -> Result<Self, String> {
        let Ok(raw) = std::env::var(READ_DENY_FILES_ENV) else {
            return Ok(Self(Vec::new()));
        };
        let files: Vec<String> = serde_json::from_str(&raw)
            .map_err(|e| format!("{READ_DENY_FILES_ENV} is not a JSON list of paths: {e}"))?;
        let mut patterns = Vec::new();
        for file in files {
            let text = fs::read_to_string(&file)
                .map_err(|e| format!("Cannot read the deny rules in {file}: {e}"))?;
            let value: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| format!("Cannot read the deny rules in {file}: {e}"))?;
            let Some(deny) = value.pointer("/permissions/deny") else {
                continue;
            };
            let deny = deny
                .as_array()
                .ok_or_else(|| format!("permissions.deny in {file} is not a list"))?;
            for rule in deny.iter().filter_map(|r| r.as_str()) {
                if let Some(spec) = rule.strip_prefix("Read(").and_then(|r| r.strip_suffix(')')) {
                    patterns.push(deny_regex(spec)?);
                }
            }
        }
        Ok(Self(patterns))
    }

    fn denies(&self, path: &Path) -> bool {
        if self.0.is_empty() {
            return false;
        }
        let text = path.to_string_lossy().replace('\\', "/");
        self.0.iter().any(|re| re.is_match(&text))
    }
}

fn deny_regex(spec: &str) -> Result<Regex, String> {
    let mut pattern = spec.replace('\\', "/");
    if let Some(rest) = pattern.strip_prefix("//") {
        pattern = rest.to_string();
    }
    pattern = pattern.replace("**", "*");
    let has_drive = {
        let bytes = pattern.as_bytes();
        bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
    };
    if !has_drive {
        pattern = format!("*{pattern}");
    }
    let body = pattern
        .split('*')
        .map(regex::escape)
        .collect::<Vec<_>>()
        .join(".*");
    Regex::new(&format!("(?is)^{body}$")).map_err(|e| format!("Deny rule Read({spec}): {e}"))
}

/// Walker with the rules shared by grep, glob and tree.
fn walker(root: &Path) -> WalkBuilder {
    let mut builder = WalkBuilder::new(root);
    builder
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .require_git(false)
        .ignore(true)
        .hidden(false)
        .follow_links(false)
        .sort_by_file_name(|a, b| a.cmp(b))
        .filter_entry(|entry| entry.file_name() != ".git");
    builder
}

/// Regular files reachable from the walker, without links and without denied files.
fn files(builder: WalkBuilder, deny: &DenyList) -> impl Iterator<Item = DirEntry> + '_ {
    builder.build().flatten().filter(move |entry| {
        !entry.path_is_symlink()
            && entry.file_type().is_some_and(|t| t.is_file())
            && !deny.denies(entry.path())
    })
}

fn modified(path: &Path) -> SystemTime {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Path as shown in results: relative to the working directory when inside it, with the
/// platform's separator throughout (a path given as `C:/x` would otherwise show as `C:/x\y`).
fn display(path: &Path, base: &Path) -> String {
    let shown = path
        .strip_prefix(base)
        .ok()
        .filter(|rel| !rel.as_os_str().is_empty())
        .unwrap_or(path)
        .display()
        .to_string();
    if cfg!(windows) {
        shown.replace('/', "\\")
    } else {
        shown
    }
}

/// Text of a file, or `None` for binary, unreadable and very large files.
fn read_text(path: &Path) -> Option<String> {
    let file = fs::File::open(path).ok()?;
    if file.metadata().ok()?.len() > MAX_SEARCHED_FILE_BYTES {
        return None;
    }
    let mut bytes = Vec::new();
    file.take(MAX_SEARCHED_FILE_BYTES)
        .read_to_end(&mut bytes)
        .ok()?;
    let (bom, body) = Bom::detect(&bytes);
    match bom {
        Bom::Utf16Le | Bom::Utf16Be => Some(bom.decode_lossy(body)),
        Bom::Utf8 | Bom::None => {
            let probe = &body[..body.len().min(BINARY_PROBE_BYTES)];
            if probe.contains(&0) {
                None
            } else {
                Some(String::from_utf8_lossy(body).into_owned())
            }
        }
    }
}

fn clip_line(line: &str) -> String {
    let line = line.strip_suffix('\r').unwrap_or(line);
    if line.chars().count() <= MAX_LINE_CHARS {
        return line.to_string();
    }
    let kept: String = line.chars().take(MAX_LINE_CHARS).collect();
    format!("{kept} [... line cut at {MAX_LINE_CHARS} characters]")
}

/// Line numbers (0-based) that contain a match.
fn matching_lines(text: &str, line_starts: &[usize], re: &Regex, multiline: bool) -> Vec<usize> {
    let line_of = |offset: usize| match line_starts.binary_search(&offset) {
        Ok(i) => i,
        Err(i) => i - 1,
    };
    if multiline {
        let mut lines = Vec::new();
        for m in re.find_iter(text) {
            let first = line_of(m.start());
            let last = line_of(m.end().saturating_sub(1).max(m.start()));
            for line in first..=last {
                if lines.last() != Some(&line) {
                    lines.push(line);
                }
            }
        }
        lines
    } else {
        text.split('\n')
            .enumerate()
            .filter(|(_, line)| re.is_match(line.strip_suffix('\r').unwrap_or(line)))
            .map(|(i, _)| i)
            .collect()
    }
}

fn line_starts(text: &str) -> Vec<usize> {
    std::iter::once(0)
        .chain(text.match_indices('\n').map(|(i, _)| i + 1))
        .filter(|&start| start <= text.len())
        .collect()
}

/// Applies offset and head_limit to entries and caps the result at the output limit.
fn page(entries: Vec<String>, offset: usize, head_limit: usize) -> String {
    let total = entries.len();
    let mut out = String::new();
    let mut shown = 0;
    let end = if head_limit == 0 {
        total
    } else {
        offset.saturating_add(head_limit).min(total)
    };
    let mut byte_capped = false;
    for entry in entries.iter().take(end).skip(offset) {
        if out.len() + entry.len() + 1 > OUTPUT_LIMIT_BYTES {
            byte_capped = true;
            break;
        }
        out.push_str(entry);
        out.push('\n');
        shown += 1;
    }
    let next = offset + shown;
    if next < total {
        let why = if byte_capped {
            format!("output limited to {OUTPUT_LIMIT_BYTES} bytes")
        } else {
            format!("head_limit {head_limit}")
        };
        out.push_str(&format!(
            "[{} more not shown ({why}); continue with offset={next}]",
            total - next
        ));
    }
    out.trim_end_matches('\n').to_string()
}

impl SearchTools {
    pub fn new() -> Self {
        Self
    }

    fn base(working_dir: Option<&Path>) -> PathBuf {
        working_dir
            .map(Path::to_path_buf)
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    pub fn grep_with_cwd(&self, params: GrepParams, working_dir: Option<&Path>) -> CallToolResult {
        let base = Self::base(working_dir);
        let root = params
            .path
            .as_deref()
            .filter(|p| !p.is_empty())
            .map(|p| resolve_path(p, Some(&base)))
            .unwrap_or_else(|| base.clone());
        if !root.exists() {
            return error(format!("Path does not exist: {}", root.display()));
        }
        let deny = match DenyList::load() {
            Ok(deny) => deny,
            Err(e) => return error(format!("Search refused: {e}")),
        };
        if deny.denies(&root) {
            return error(format!(
                "Search refused: {} is blocked by a Read deny rule.",
                root.display()
            ));
        }

        let multiline = params.multiline.unwrap_or(false);
        let re = match RegexBuilder::new(&params.pattern)
            .case_insensitive(params.case_insensitive.unwrap_or(false))
            .multi_line(true)
            .dot_matches_new_line(multiline)
            .build()
        {
            Ok(re) => re,
            Err(e) => return error(format!("Invalid regular expression: {e}")),
        };

        let mut builder = walker(&root);
        if let Some(glob) = params.glob.as_deref().filter(|g| !g.is_empty()) {
            let mut overrides = OverrideBuilder::new(&root);
            for part in split_globs(glob) {
                if let Err(e) = overrides.add(&part) {
                    return error(format!("Invalid glob {part}: {e}"));
                }
            }
            match overrides.build() {
                Ok(o) => {
                    builder.overrides(o);
                }
                Err(e) => return error(format!("Invalid glob {glob}: {e}")),
            }
        }
        if let Some(file_type) = params.file_type.as_deref().filter(|t| !t.is_empty()) {
            let mut types = TypesBuilder::new();
            types.add_defaults();
            types.select(file_type);
            match types.build() {
                Ok(t) => {
                    builder.types(t);
                }
                Err(e) => return error(format!("Unknown file type {file_type}: {e}")),
            }
        }

        let context = params.context_short.or(params.context);
        let before = params.before.or(context).unwrap_or(0);
        let after = params.after.or(context).unwrap_or(0);
        let numbers = params.line_numbers.unwrap_or(true);
        let offset = params.offset.unwrap_or(0);
        let head_limit = params.head_limit.unwrap_or(DEFAULT_HEAD_LIMIT);

        let mut hits: Vec<(PathBuf, String, Vec<usize>)> = Vec::new();
        for entry in files(builder, &deny) {
            let Some(text) = read_text(entry.path()) else {
                continue;
            };
            let starts = line_starts(&text);
            let lines = matching_lines(&text, &starts, &re, multiline);
            if !lines.is_empty() {
                hits.push((entry.path().to_path_buf(), text, lines));
            }
        }
        if hits.is_empty() {
            return success("No matches found");
        }

        let entries = match params.output_mode {
            OutputMode::FilesWithMatches => {
                let mut paths: Vec<(SystemTime, String)> = hits
                    .iter()
                    .map(|(path, _, _)| (modified(path), display(path, &base)))
                    .collect();
                paths.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                let mut entries = vec![format!(
                    "Found {} file{}",
                    paths.len(),
                    if paths.len() == 1 { "" } else { "s" }
                )];
                let listed = page(
                    paths.into_iter().map(|(_, p)| p).collect(),
                    offset,
                    head_limit,
                );
                entries.push(listed);
                return success(entries.join("\n"));
            }
            OutputMode::Count => {
                let total: usize = hits.iter().map(|(_, _, l)| l.len()).sum();
                let lines: Vec<String> = hits
                    .iter()
                    .map(|(path, _, l)| format!("{}:{}", display(path, &base), l.len()))
                    .collect();
                let files = lines.len();
                let mut out = page(lines, offset, head_limit);
                out.push_str(&format!(
                    "\n\nFound {total} matching line{} in {files} file{}.",
                    if total == 1 { "" } else { "s" },
                    if files == 1 { "" } else { "s" }
                ));
                return success(out);
            }
            OutputMode::Content => content_lines(&hits, &base, before, after, numbers),
        };
        success(page(entries, offset, head_limit))
    }

    pub fn glob_with_cwd(&self, params: GlobParams, working_dir: Option<&Path>) -> CallToolResult {
        let base = Self::base(working_dir);
        let root = params
            .path
            .as_deref()
            .filter(|p| !p.is_empty())
            .map(|p| resolve_path(p, Some(&base)))
            .unwrap_or_else(|| base.clone());
        if !root.is_dir() {
            return error(format!("Not a directory: {}", root.display()));
        }
        let deny = match DenyList::load() {
            Ok(deny) => deny,
            Err(e) => return error(format!("Search refused: {e}")),
        };
        if deny.denies(&root) {
            return error(format!(
                "Search refused: {} is blocked by a Read deny rule.",
                root.display()
            ));
        }
        let mut overrides = OverrideBuilder::new(&root);
        for part in split_globs(&params.pattern) {
            if let Err(e) = overrides.add(&part) {
                return error(format!("Invalid glob {part}: {e}"));
            }
        }
        let overrides = match overrides.build() {
            Ok(o) => o,
            Err(e) => return error(format!("Invalid glob {}: {e}", params.pattern)),
        };
        let mut builder = walker(&root);
        builder.overrides(overrides);

        let mut found: Vec<(SystemTime, String)> = files(builder, &deny)
            .map(|entry| (modified(entry.path()), display(entry.path(), &base)))
            .collect();
        if found.is_empty() {
            return success("No files found");
        }
        found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let total = found.len();
        let mut out: Vec<String> = found.into_iter().take(GLOB_LIMIT).map(|(_, p)| p).collect();
        if total > GLOB_LIMIT {
            out.push(format!(
                "(Results are truncated: {GLOB_LIMIT} of {total} files shown. Use a narrower pattern or path.)"
            ));
        }
        success(page(out, 0, 0))
    }
}

/// `a,b` at the top level are two globs only inside braces they stay one (`*.{ts,tsx}`).
fn split_globs(glob: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut current = String::new();
    for c in glob.chars() {
        match c {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            ',' | ' ' if depth == 0 => {
                if !current.trim().is_empty() {
                    parts.push(current.trim().to_string());
                }
                current.clear();
                continue;
            }
            _ => {}
        }
        current.push(c);
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    parts
}

/// Matching lines with context, groups separated by `--` like ripgrep.
fn content_lines(
    hits: &[(PathBuf, String, Vec<usize>)],
    base: &Path,
    before: usize,
    after: usize,
    numbers: bool,
) -> Vec<String> {
    let mut out = Vec::new();
    let with_context = before > 0 || after > 0;
    for (path, text, lines) in hits {
        let shown = display(path, base);
        let all: Vec<&str> = text.split('\n').collect();
        let last_line = all.len().saturating_sub(1);
        let mut previous_end: Option<usize> = None;
        let mut i = 0;
        while i < lines.len() {
            let start = lines[i].saturating_sub(before);
            let mut end = (lines[i] + after).min(last_line);
            let mut group = vec![lines[i]];
            while i + 1 < lines.len() && lines[i + 1].saturating_sub(before) <= end + 1 {
                i += 1;
                group.push(lines[i]);
                end = (lines[i] + after).min(last_line);
            }
            if with_context && (previous_end.is_some() || !out.is_empty()) {
                out.push("--".to_string());
            }
            let start = match previous_end {
                Some(p) if start <= p => p + 1,
                _ => start,
            };
            for n in start..=end {
                let Some(line) = all.get(n) else { break };
                if n == last_line && line.is_empty() {
                    break;
                }
                let sep = if group.contains(&n) { ':' } else { '-' };
                if numbers {
                    out.push(format!("{shown}{sep}{}{sep}{}", n + 1, clip_line(line)));
                } else {
                    out.push(format!("{shown}{sep}{}", clip_line(line)));
                }
            }
            previous_end = Some(end);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn text(result: &CallToolResult) -> String {
        match &result.content[0] {
            ContentBlock::Text(t) => t.text.clone(),
            _ => panic!("expected text"),
        }
    }

    fn grep(dir: &Path, json: serde_json::Value) -> CallToolResult {
        let params: GrepParams = serde_json::from_value(json).unwrap();
        SearchTools::new().grep_with_cwd(params, Some(dir))
    }

    fn setup() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::create_dir_all(dir.path().join(".claude")).unwrap();
        fs::create_dir_all(dir.path().join(".git")).unwrap();
        fs::write(
            dir.path().join("src/a.rs"),
            "fn alpha() {}\nlet x = 1;\nfn beta() {}\nlet y = 2;\nlet z = 3;\n",
        )
        .unwrap();
        fs::write(dir.path().join("src/b.mjs"), "export const alpha = 1;\n").unwrap();
        fs::write(dir.path().join(".claude/notes.md"), "alpha in hidden\n").unwrap();
        fs::write(dir.path().join(".git/config"), "alpha in git\n").unwrap();
        fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        fs::write(dir.path().join("ignored.txt"), "alpha ignored\n").unwrap();
        fs::write(dir.path().join("blob.bin"), b"alpha\0\x01\x02").unwrap();
        dir
    }

    #[test]
    fn files_mode_respects_ignores_and_includes_hidden() {
        let dir = setup();
        let out = text(&grep(dir.path(), serde_json::json!({"pattern": "alpha"})));
        assert!(out.starts_with("Found 3 files"), "{out}");
        assert!(out.contains("a.rs") && out.contains("b.mjs") && out.contains("notes.md"));
        assert!(!out.contains("config") && !out.contains("ignored") && !out.contains("blob"));
    }

    #[test]
    fn content_mode_shows_lines_and_context() {
        let dir = setup();
        let out = text(&grep(
            dir.path(),
            serde_json::json!({"pattern": "^fn", "output_mode": "content", "glob": "*.rs", "-A": 1}),
        ));
        let a = Path::new("src").join("a.rs").display().to_string();
        assert_eq!(
            out,
            format!(
                "{a}:1:fn alpha() {{}}\n{a}-2-let x = 1;\n{a}:3:fn beta() {{}}\n{a}-4-let y = 2;"
            )
        );
    }

    #[test]
    fn separate_groups_get_a_separator() {
        let dir = setup();
        let out = text(&grep(
            dir.path(),
            serde_json::json!({"pattern": "alpha|z =", "output_mode": "content", "type": "rust", "-B": 0, "-A": 0, "-C": 0}),
        ));
        assert!(!out.contains("--"), "no context, no separator: {out}");
        let out = text(&grep(
            dir.path(),
            serde_json::json!({"pattern": "alpha|z =", "output_mode": "content", "type": "rust", "context": 1}),
        ));
        assert!(out.contains("\n--\n"), "{out}");
    }

    #[test]
    fn count_case_insensitive_and_limits() {
        let dir = setup();
        let out = text(&grep(
            dir.path(),
            serde_json::json!({"pattern": "LET", "-i": true, "output_mode": "count"}),
        ));
        assert!(out.contains("a.rs:3"), "{out}");
        assert!(out.contains("Found 3 matching lines in 1 file."), "{out}");

        let out = text(&grep(
            dir.path(),
            serde_json::json!({"pattern": "let", "output_mode": "content", "head_limit": 1, "-n": false}),
        ));
        let a = Path::new("src").join("a.rs").display().to_string();
        assert!(out.starts_with(&format!("{a}:let x = 1;\n")), "{out}");
        assert!(out.contains("continue with offset=1"), "{out}");
        let out = text(&grep(
            dir.path(),
            serde_json::json!({"pattern": "let", "output_mode": "content", "head_limit": 1, "offset": 2}),
        ));
        assert_eq!(out, format!("{a}:5:let z = 3;"));
    }

    #[test]
    fn multiline_and_errors() {
        let dir = setup();
        let out = text(&grep(
            dir.path(),
            serde_json::json!({"pattern": "alpha.*beta", "multiline": true, "output_mode": "content"}),
        ));
        assert!(
            out.contains(":1:fn alpha") && out.contains(":3:fn beta"),
            "{out}"
        );
        assert!(out.contains(":2:let x"), "{out}");

        let bad = grep(dir.path(), serde_json::json!({"pattern": "("}));
        assert_eq!(bad.is_error, Some(true));
        assert!(text(&bad).contains("Invalid regular expression"));
        let missing = grep(
            dir.path(),
            serde_json::json!({"pattern": "x", "path": "nope"}),
        );
        assert_eq!(missing.is_error, Some(true));
        let none = text(&grep(
            dir.path(),
            serde_json::json!({"pattern": "nothing-here"}),
        ));
        assert_eq!(none, "No matches found");
    }

    #[test]
    fn utf16_files_are_searched() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("w.ps1"),
            Bom::Utf16Le.encode("Write-Host alpha\r\n"),
        )
        .unwrap();
        let out = text(&grep(
            dir.path(),
            serde_json::json!({"pattern": "alpha", "output_mode": "content"}),
        ));
        assert_eq!(out, "w.ps1:1:Write-Host alpha");
    }

    #[test]
    fn glob_finds_files_newest_first() {
        let dir = setup();
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(dir.path().join("src/c.rs"), "new\n").unwrap();
        let params = GlobParams {
            pattern: "**/*.rs".into(),
            path: None,
        };
        let out = text(&SearchTools::new().glob_with_cwd(params, Some(dir.path())));
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2, "{out}");
        assert!(lines[0].ends_with("c.rs"), "{out}");
        let params = GlobParams {
            pattern: "*.{md,mjs}".into(),
            path: None,
        };
        let out = text(&SearchTools::new().glob_with_cwd(params, Some(dir.path())));
        assert!(out.contains("notes.md") && out.contains("b.mjs"), "{out}");
    }

    #[test]
    fn deny_patterns_follow_the_server_rules() {
        let re = deny_regex("**/secrets.yaml").unwrap();
        assert!(re.is_match("C:/Users/x/AppData/Roaming/Block/goose/config/SECRETS.yaml"));
        assert!(!re.is_match("C:/Users/x/secrets.yaml.bak"));
        let abs = deny_regex("//C:/Users/x/private/**").unwrap();
        assert!(abs.is_match("C:/Users/x/private/a/b.txt"));
        assert!(!abs.is_match("D:/C:/Users/x/private/a"));
        let list = DenyList(vec![deny_regex("**/.env").unwrap()]);
        assert!(list.denies(Path::new(r"C:\project\.env")));
        assert!(!list.denies(Path::new(r"C:\project\.env.example")));
    }

    #[cfg(windows)]
    #[test]
    fn paths_outside_the_working_dir_use_one_separator() {
        let shown = display(
            &Path::new("C:/data/project").join(".claude").join("b.md"),
            Path::new(r"C:\elsewhere"),
        );
        assert_eq!(shown, r"C:\data\project\.claude\b.md");
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_not_followed() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.txt"), "alpha secret\n").unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("secret.txt"),
            dir.path().join("link.txt"),
        )
        .unwrap();
        let out = text(&grep(dir.path(), serde_json::json!({"pattern": "alpha"})));
        assert_eq!(out, "No matches found");
    }
}
