//! notebook_edit: change, insert or delete one cell of a Jupyter notebook (.ipynb), like Claude
//! Code's NotebookEdit. The rest of the notebook stays as it is: other cells with their outputs,
//! metadata, the file's encoding, indentation and line endings.

use std::fs;
use std::path::Path;

use super::edit::resolve_path;
use super::encoding::Bom;
use rand::RngExt;
use rmcp::model::{Annotations, CallToolResult, ContentBlock, TextContent};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::ser::PrettyFormatter;
use serde_json::{json, Map, Value};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum NotebookEditMode {
    /// Replace the source of the cell `cell_id` (its outputs are cleared).
    #[default]
    Replace,
    /// Insert a new cell after the cell `cell_id`, or at the start when `cell_id` is left out.
    Insert,
    /// Delete the cell `cell_id`.
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum NotebookCellType {
    Code,
    Markdown,
}

impl NotebookCellType {
    fn name(self) -> &'static str {
        match self {
            NotebookCellType::Code => "code",
            NotebookCellType::Markdown => "markdown",
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct NotebookEditParams {
    /// Path of the notebook (.ipynb file).
    pub path: String,
    /// What to do: replace (default), insert or delete.
    #[serde(default)]
    pub edit_mode: NotebookEditMode,
    /// The cell to change: its id as shown by read (`id ...`), or its number as shown by read
    /// (`Cell 3`, also written `cell-3`). With insert the new cell goes after this cell; left out,
    /// it goes to the start of the notebook.
    #[serde(default)]
    pub cell_id: Option<String>,
    /// The new source of the cell. Required for replace and insert.
    #[serde(default)]
    pub new_source: Option<String>,
    /// code or markdown. Required for insert; with replace it changes the type of the cell.
    #[serde(default)]
    pub cell_type: Option<NotebookCellType>,
}

pub struct NotebookTools;

fn visible_text(text: impl Into<String>) -> ContentBlock {
    ContentBlock::Text(
        TextContent::new(text).with_annotations(Annotations::default().with_priority(0.0)),
    )
}

fn fail(text: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![visible_text(text)])
}

impl NotebookTools {
    pub fn new() -> Self {
        Self
    }

    pub fn notebook_edit_with_cwd(
        &self,
        params: NotebookEditParams,
        working_dir: Option<&Path>,
    ) -> CallToolResult {
        let path = resolve_path(&params.path, working_dir);
        if !params.path.to_lowercase().ends_with(".ipynb") {
            return fail(format!(
                "{} is not a Jupyter notebook (.ipynb). Use edit for other files.",
                params.path
            ));
        }
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => return fail(format!("Failed to read {}: {}", params.path, error)),
        };
        let (bom, body) = Bom::detect(&bytes);
        let text = match bom.decode_strict(body) {
            Ok(text) => text,
            Err(reason) => {
                return fail(format!(
                    "Failed to read {}: {}. notebook_edit would corrupt it.",
                    params.path, reason
                ))
            }
        };
        let mut notebook: Value = match serde_json::from_str(&text) {
            Ok(value) => value,
            Err(error) => return fail(format!("{} is not a valid notebook: {error}", params.path)),
        };
        let message = match apply(&mut notebook, &params) {
            Ok(message) => message,
            Err(message) => return fail(message),
        };
        let out = match serialize_like(&notebook, &text) {
            Ok(out) => out,
            Err(error) => return fail(format!("Failed to serialize the notebook: {error}")),
        };
        match fs::write(&path, bom.encode(&out)) {
            Ok(()) => {
                CallToolResult::success(vec![visible_text(format!("{message} in {}", params.path))])
            }
            Err(error) => fail(format!("Failed to write {}: {}", params.path, error)),
        }
    }
}

impl Default for NotebookTools {
    fn default() -> Self {
        Self::new()
    }
}

/// The cell's source as the list of lines Jupyter stores (every line but the last keeps its `\n`).
fn source_lines(text: &str) -> Value {
    Value::Array(
        text.split_inclusive('\n')
            .map(|line| Value::String(line.to_string()))
            .collect(),
    )
}

/// Position of the cell: its id, else its number as shown by read (`3` or `cell-3`, counted from 1).
fn find_cell(cells: &[Value], cell_id: &str) -> Result<usize, String> {
    let wanted = cell_id.trim();
    if let Some(index) = cells
        .iter()
        .position(|cell| cell["id"].as_str() == Some(wanted))
    {
        return Ok(index);
    }
    let number = wanted
        .strip_prefix("cell-")
        .or_else(|| wanted.strip_prefix("Cell "))
        .unwrap_or(wanted);
    if let Ok(number) = number.trim().parse::<usize>() {
        if (1..=cells.len()).contains(&number) {
            return Ok(number - 1);
        }
    }
    Err(format!(
        "No cell with id or number \"{cell_id}\". The notebook has {} cells (numbers 1 to {}); read shows the ids.",
        cells.len(),
        cells.len()
    ))
}

/// Notebook format 4.5 and later gives every cell an id.
fn cells_have_ids(notebook: &Value) -> bool {
    let major = notebook["nbformat"].as_u64().unwrap_or(4);
    let minor = notebook["nbformat_minor"].as_u64().unwrap_or(0);
    major > 4 || (major == 4 && minor >= 5)
}

fn new_cell_id(cells: &[Value]) -> String {
    loop {
        let id = format!("{:08x}", rand::rng().random::<u32>());
        if !cells.iter().any(|cell| cell["id"].as_str() == Some(&id)) {
            return id;
        }
    }
}

fn set_type(cell: &mut Map<String, Value>, kind: &str) {
    cell.insert("cell_type".to_string(), json!(kind));
    if kind == "code" {
        cell.insert("execution_count".to_string(), Value::Null);
        cell.insert("outputs".to_string(), json!([]));
    } else {
        cell.remove("execution_count");
        cell.remove("outputs");
    }
}

/// Changes the notebook in place; the answer says what was done.
fn apply(notebook: &mut Value, params: &NotebookEditParams) -> Result<String, String> {
    let ids = cells_have_ids(notebook);
    let Some(cells) = notebook.get_mut("cells").and_then(Value::as_array_mut) else {
        return Err(format!(
            "{} is not a valid notebook: no cells.",
            params.path
        ));
    };
    let need_cell = || {
        params
            .cell_id
            .as_deref()
            .ok_or_else(|| "cell_id is required for replace and delete.".to_string())
    };
    let need_source = || {
        params
            .new_source
            .as_deref()
            .ok_or_else(|| "new_source is required for replace and insert.".to_string())
    };

    match params.edit_mode {
        NotebookEditMode::Replace => {
            let index = find_cell(cells, need_cell()?)?;
            let source = need_source()?;
            let Some(cell) = cells[index].as_object_mut() else {
                return Err(format!("Cell {} is not a valid cell.", index + 1));
            };
            cell.insert("source".to_string(), source_lines(source));
            let old_kind = cell
                .get("cell_type")
                .and_then(Value::as_str)
                .unwrap_or("code")
                .to_string();
            let kind = params.cell_type.map_or(old_kind.as_str(), |t| t.name());
            let mut note = String::new();
            if kind != old_kind {
                set_type(cell, kind);
                note = format!(", now a {kind} cell");
            } else if kind == "code" {
                // The old outputs belong to the old source.
                cell.insert("execution_count".to_string(), Value::Null);
                cell.insert("outputs".to_string(), json!([]));
                note = ", outputs cleared".to_string();
            }
            Ok(format!("Replaced the source of cell {}{note}", index + 1))
        }
        NotebookEditMode::Insert => {
            let source = need_source()?;
            let Some(kind) = params.cell_type else {
                return Err("cell_type (code or markdown) is required for insert.".to_string());
            };
            let at = match params.cell_id.as_deref() {
                Some(id) => find_cell(cells, id)? + 1,
                None => 0,
            };
            let mut cell = Map::new();
            cell.insert("cell_type".to_string(), json!(kind.name()));
            if kind == NotebookCellType::Code {
                cell.insert("execution_count".to_string(), Value::Null);
            }
            let id = ids.then(|| new_cell_id(cells));
            if let Some(id) = &id {
                cell.insert("id".to_string(), json!(id));
            }
            cell.insert("metadata".to_string(), json!({}));
            if kind == NotebookCellType::Code {
                cell.insert("outputs".to_string(), json!([]));
            }
            cell.insert("source".to_string(), source_lines(source));
            cells.insert(at, Value::Object(cell));
            let id = id.map(|id| format!(", id {id}")).unwrap_or_default();
            Ok(format!(
                "Inserted a {} cell as cell {}{id}",
                kind.name(),
                at + 1
            ))
        }
        NotebookEditMode::Delete => {
            let index = find_cell(cells, need_cell()?)?;
            let removed = cells.remove(index);
            let kind = removed["cell_type"].as_str().unwrap_or("code").to_string();
            Ok(format!("Deleted the {kind} cell {}", index + 1))
        }
    }
}

/// The notebook as text, laid out like the original: its indentation (Jupyter uses one space,
/// editors often two), line endings and a final newline.
fn serialize_like(notebook: &Value, original: &str) -> Result<String, serde_json::Error> {
    let indent: String = original
        .lines()
        .nth(1)
        .map(|line| {
            line.chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .collect()
        })
        .filter(|indent: &String| !indent.is_empty())
        .unwrap_or_else(|| " ".to_string());
    let mut out = Vec::new();
    let formatter = PrettyFormatter::with_indent(indent.as_bytes());
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, formatter);
    serde::Serialize::serialize(notebook, &mut serializer)?;
    let mut text = String::from_utf8(out).unwrap_or_default();
    if original.ends_with('\n') {
        text.push('\n');
    }
    if original.contains("\r\n") {
        text = text.replace('\n', "\r\n");
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// A notebook as Jupyter writes it: sorted keys, one space of indentation, final newline.
    const NOTEBOOK: &str = r##"{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "aaaa1111",
   "metadata": {},
   "source": [
    "# Title\n",
    "text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "bbbb2222",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3\n"
     ]
    }
   ],
   "source": [
    "print(1 + 2)"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
"##;

    fn text_of(result: &CallToolResult) -> &str {
        match &result.content[0] {
            ContentBlock::Text(text) => &text.text,
            _ => panic!("expected text"),
        }
    }

    fn params(path: &Path, mode: NotebookEditMode) -> NotebookEditParams {
        NotebookEditParams {
            path: path.to_string_lossy().to_string(),
            edit_mode: mode,
            cell_id: None,
            new_source: None,
            cell_type: None,
        }
    }

    fn setup(content: &str) -> (TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("n.ipynb");
        fs::write(&path, content).unwrap();
        (dir, path)
    }

    fn cells_of(path: &Path) -> Vec<Value> {
        let value: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        value["cells"].as_array().unwrap().clone()
    }

    #[test]
    fn replace_changes_one_cell_and_leaves_the_rest_byte_for_byte() {
        let (_dir, path) = setup(NOTEBOOK);
        let mut p = params(&path, NotebookEditMode::Replace);
        p.cell_id = Some("aaaa1111".to_string());
        p.new_source = Some("# Title\ntext".to_string());
        // Same source as before: the file must come back unchanged.
        let result = NotebookTools::new().notebook_edit_with_cwd(p, None);
        assert_eq!(result.is_error, Some(false), "{}", text_of(&result));
        assert_eq!(fs::read_to_string(&path).unwrap(), NOTEBOOK);
    }

    #[test]
    fn replace_code_cell_clears_outputs_and_keeps_the_other_cell() {
        let (_dir, path) = setup(NOTEBOOK);
        let mut p = params(&path, NotebookEditMode::Replace);
        p.cell_id = Some("bbbb2222".to_string());
        p.new_source = Some("a = 1\nprint(a)".to_string());
        let result = NotebookTools::new().notebook_edit_with_cwd(p, None);
        assert!(text_of(&result).contains("outputs cleared"));
        let cells = cells_of(&path);
        assert_eq!(cells[1]["source"], json!(["a = 1\n", "print(a)"]));
        assert_eq!(cells[1]["outputs"], json!([]));
        assert_eq!(cells[1]["execution_count"], Value::Null);
        assert_eq!(cells[0]["source"], json!(["# Title\n", "text"]));
    }

    #[test]
    fn cell_can_be_found_by_number() {
        let (_dir, path) = setup(NOTEBOOK);
        for name in ["2", "cell-2"] {
            let mut p = params(&path, NotebookEditMode::Replace);
            p.cell_id = Some(name.to_string());
            p.new_source = Some(format!("x = '{name}'"));
            let result = NotebookTools::new().notebook_edit_with_cwd(p, None);
            assert_eq!(result.is_error, Some(false), "{}", text_of(&result));
            assert_eq!(
                cells_of(&path)[1]["source"],
                json!([format!("x = '{name}'")])
            );
        }
    }

    #[test]
    fn replace_can_change_the_cell_type() {
        let (_dir, path) = setup(NOTEBOOK);
        let mut p = params(&path, NotebookEditMode::Replace);
        p.cell_id = Some("bbbb2222".to_string());
        p.new_source = Some("Now text".to_string());
        p.cell_type = Some(NotebookCellType::Markdown);
        NotebookTools::new().notebook_edit_with_cwd(p, None);
        let cell = &cells_of(&path)[1];
        assert_eq!(cell["cell_type"], "markdown");
        assert!(cell.get("outputs").is_none() && cell.get("execution_count").is_none());

        let mut p = params(&path, NotebookEditMode::Replace);
        p.cell_id = Some("bbbb2222".to_string());
        p.new_source = Some("print(1)".to_string());
        p.cell_type = Some(NotebookCellType::Code);
        NotebookTools::new().notebook_edit_with_cwd(p, None);
        let cell = &cells_of(&path)[1];
        assert_eq!(cell["cell_type"], "code");
        assert_eq!(cell["outputs"], json!([]));
        assert_eq!(cell["execution_count"], Value::Null);
    }

    #[test]
    fn insert_puts_a_new_cell_after_the_given_one_with_a_fresh_id() {
        let (_dir, path) = setup(NOTEBOOK);
        let mut p = params(&path, NotebookEditMode::Insert);
        p.cell_id = Some("aaaa1111".to_string());
        p.new_source = Some("import os".to_string());
        p.cell_type = Some(NotebookCellType::Code);
        let result = NotebookTools::new().notebook_edit_with_cwd(p, None);
        assert!(
            text_of(&result).contains("as cell 2"),
            "{}",
            text_of(&result)
        );
        let cells = cells_of(&path);
        assert_eq!(cells.len(), 3);
        assert_eq!(cells[1]["source"], json!(["import os"]));
        assert_eq!(cells[1]["cell_type"], "code");
        let id = cells[1]["id"].as_str().unwrap();
        assert_eq!(id.len(), 8);
        assert_ne!(id, "aaaa1111");
        assert_eq!(cells[2]["id"], "bbbb2222");
    }

    #[test]
    fn insert_without_cell_id_goes_to_the_start_and_markdown_has_no_outputs() {
        let (_dir, path) = setup(NOTEBOOK);
        let mut p = params(&path, NotebookEditMode::Insert);
        p.new_source = Some("Intro".to_string());
        p.cell_type = Some(NotebookCellType::Markdown);
        NotebookTools::new().notebook_edit_with_cwd(p, None);
        let cells = cells_of(&path);
        assert_eq!(cells[0]["source"], json!(["Intro"]));
        assert!(cells[0].get("outputs").is_none());
        assert_eq!(cells[1]["id"], "aaaa1111");
    }

    #[test]
    fn older_notebook_format_gets_no_cell_ids() {
        let old = NOTEBOOK.replace("\"nbformat_minor\": 5", "\"nbformat_minor\": 2");
        let (_dir, path) = setup(&old);
        let mut p = params(&path, NotebookEditMode::Insert);
        p.new_source = Some("1".to_string());
        p.cell_type = Some(NotebookCellType::Code);
        NotebookTools::new().notebook_edit_with_cwd(p, None);
        assert!(cells_of(&path)[0].get("id").is_none());
    }

    #[test]
    fn delete_removes_the_cell() {
        let (_dir, path) = setup(NOTEBOOK);
        let mut p = params(&path, NotebookEditMode::Delete);
        p.cell_id = Some("aaaa1111".to_string());
        let result = NotebookTools::new().notebook_edit_with_cwd(p, None);
        assert!(text_of(&result).contains("Deleted the markdown cell 1"));
        let cells = cells_of(&path);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0]["id"], "bbbb2222");
    }

    #[test]
    fn mistakes_are_reported_and_leave_the_file_alone() {
        let (_dir, path) = setup(NOTEBOOK);
        let tools = NotebookTools::new();
        let run = |edit: &dyn Fn(&mut NotebookEditParams)| {
            let mut p = params(&path, NotebookEditMode::Replace);
            edit(&mut p);
            tools.notebook_edit_with_cwd(p, None)
        };
        let unknown = run(&|p| {
            p.cell_id = Some("nope".to_string());
            p.new_source = Some("x".to_string());
        });
        assert_eq!(unknown.is_error, Some(true));
        assert!(text_of(&unknown).contains("2 cells"));
        let out_of_range = run(&|p| {
            p.cell_id = Some("3".to_string());
            p.new_source = Some("x".to_string());
        });
        assert_eq!(out_of_range.is_error, Some(true));
        assert!(text_of(&run(&|p| p.cell_id = Some("1".to_string())))
            .contains("new_source is required"));
        assert!(text_of(&run(&|p| p.new_source = Some("x".to_string())))
            .contains("cell_id is required"));
        let no_type = tools.notebook_edit_with_cwd(
            NotebookEditParams {
                new_source: Some("x".to_string()),
                ..params(&path, NotebookEditMode::Insert)
            },
            None,
        );
        assert!(text_of(&no_type).contains("cell_type"));
        assert_eq!(fs::read_to_string(&path).unwrap(), NOTEBOOK);
    }

    #[test]
    fn rejects_other_files_and_broken_notebooks() {
        let dir = tempfile::tempdir().unwrap();
        let tools = NotebookTools::new();
        let txt = dir.path().join("a.txt");
        fs::write(&txt, "x").unwrap();
        let result = tools.notebook_edit_with_cwd(params(&txt, NotebookEditMode::Delete), None);
        assert!(text_of(&result).contains("not a Jupyter notebook"));

        let broken = dir.path().join("b.ipynb");
        fs::write(&broken, "{").unwrap();
        let result = tools.notebook_edit_with_cwd(params(&broken, NotebookEditMode::Delete), None);
        assert!(text_of(&result).contains("not a valid notebook"));

        let no_cells = dir.path().join("c.ipynb");
        fs::write(&no_cells, "{}").unwrap();
        let result =
            tools.notebook_edit_with_cwd(params(&no_cells, NotebookEditMode::Delete), None);
        assert!(text_of(&result).contains("no cells"));

        let missing = dir.path().join("missing.ipynb");
        let result = tools.notebook_edit_with_cwd(params(&missing, NotebookEditMode::Delete), None);
        assert!(text_of(&result).contains("Failed to read"));
    }

    #[test]
    fn indentation_line_endings_and_bom_stay() {
        let two: String = NOTEBOOK
            .lines()
            .map(|line| {
                let spaces = line.len() - line.trim_start().len();
                format!("{}{}\n", " ".repeat(spaces * 2), line.trim_start())
            })
            .collect();
        let crlf = two.replace('\n', "\r\n");
        let (_dir, path) = setup(&crlf);
        let mut p = params(&path, NotebookEditMode::Delete);
        p.cell_id = Some("bbbb2222".to_string());
        NotebookTools::new().notebook_edit_with_cwd(p, None);
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("\r\n  \"cells\""), "{text}");
        assert!(!text.replace("\r\n", "").contains('\n'));
        assert!(text.ends_with("}\r\n"));

        // UTF-16 with BOM stays UTF-16.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("u.ipynb");
        fs::write(&path, Bom::Utf16Le.encode(NOTEBOOK)).unwrap();
        let mut p = params(&path, NotebookEditMode::Delete);
        p.cell_id = Some("aaaa1111".to_string());
        let result = NotebookTools::new().notebook_edit_with_cwd(p, None);
        assert_eq!(result.is_error, Some(false), "{}", text_of(&result));
        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[..2], &[0xFF, 0xFE]);
        let (bom, body) = Bom::detect(&bytes);
        let text = bom.decode_strict(body).unwrap();
        assert!(text.contains("bbbb2222") && !text.contains("aaaa1111"));
    }

    #[test]
    fn relative_path_uses_the_working_directory() {
        let (dir, _path) = setup(NOTEBOOK);
        let mut p = params(Path::new("n.ipynb"), NotebookEditMode::Delete);
        p.cell_id = Some("1".to_string());
        let result = NotebookTools::new().notebook_edit_with_cwd(p, Some(dir.path()));
        assert_eq!(result.is_error, Some(false), "{}", text_of(&result));
        assert_eq!(cells_of(&dir.path().join("n.ipynb")).len(), 1);
    }

    #[test]
    fn non_ascii_text_is_written_as_it_is() {
        let (_dir, path) = setup(NOTEBOOK);
        let mut p = params(&path, NotebookEditMode::Replace);
        p.cell_id = Some("aaaa1111".to_string());
        p.new_source = Some("Größe: 5 µm".to_string());
        NotebookTools::new().notebook_edit_with_cwd(p, None);
        assert!(fs::read_to_string(&path).unwrap().contains("Größe: 5 µm"));
    }
}
