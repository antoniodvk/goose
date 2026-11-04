use std::fs;
use std::path::Path;

use rmcp::model::{Annotations, CallToolResult, ContentBlock, TextContent};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use super::edit::resolve_path;
use super::encoding::Bom;

/// Lines returned when the caller gives no limit (same default as Claude Code's Read).
const DEFAULT_LINE_LIMIT: usize = 2000;
/// Characters kept per line; the rest is cut and marked.
const MAX_LINE_CHARS: usize = 2000;
/// Upper bound of one read result (about 25k tokens, like Claude Code's output cap).
pub const READ_OUTPUT_LIMIT_BYTES: usize = 100_000;
const MAX_FILE_BYTES: u64 = 100 * 1024 * 1024;
const MAX_PDF_PAGES_PER_REQUEST: usize = 20;
const PDF_PAGES_WITHOUT_RANGE: usize = 10;
const PDF_PAGE_DECOMPRESS_LIMIT: usize = 64 * 1024 * 1024;
const IMAGE_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "gif", "webp"];

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FileReadParams {
    /// Absolute path to the file to read.
    pub path: String,
    /// Line number to start reading from (1-based). Default: 1.
    #[schemars(range(min = 1))]
    pub line: Option<u32>,
    /// Maximum number of lines to read. Default: 2000.
    #[schemars(range(min = 1))]
    pub limit: Option<u32>,
    /// Page range for PDF files, for example "3", "1-5", "2-" or "1,4-6".
    /// At most 20 pages per request; PDFs with more than 10 pages need this.
    pub pages: Option<String>,
}

pub struct ReadTool;

fn visible_text(text: impl Into<String>) -> ContentBlock {
    ContentBlock::Text(
        TextContent::new(text).with_annotations(Annotations::default().with_priority(0.0)),
    )
}

fn ok(text: impl Into<String>) -> CallToolResult {
    CallToolResult::success(vec![visible_text(text)])
}

fn fail(text: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![visible_text(text)])
}

impl ReadTool {
    pub fn new() -> Self {
        Self
    }

    pub fn file_read_with_cwd(
        &self,
        params: FileReadParams,
        working_dir: Option<&Path>,
    ) -> CallToolResult {
        let path = resolve_path(&params.path, working_dir);

        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let cwd = working_dir
                    .map(Path::to_path_buf)
                    .or_else(|| std::env::current_dir().ok());
                return fail(match cwd {
                    Some(cwd) => format!(
                        "File does not exist: {}. Current working directory: {}",
                        params.path,
                        cwd.display()
                    ),
                    None => format!("File does not exist: {}", params.path),
                });
            }
            Err(error) => return fail(format!("Failed to read {}: {}", params.path, error)),
        };
        if metadata.is_dir() {
            return fail(format!(
                "{} is a directory, not a file. Use tree to list a directory.",
                params.path
            ));
        }
        if metadata.len() > MAX_FILE_BYTES {
            return fail(format!(
                "{} is too large to read ({} MB, limit {} MB). Search it with a shell command instead.",
                params.path,
                metadata.len() / (1024 * 1024),
                MAX_FILE_BYTES / (1024 * 1024)
            ));
        }
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => return fail(format!("Failed to read {}: {}", params.path, error)),
        };

        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        if extension == "pdf" {
            return read_pdf(&params.path, &bytes, params.pages.as_deref());
        }
        if extension == "ipynb" {
            return read_notebook(&params.path, &bytes);
        }
        if IMAGE_EXTENSIONS.contains(&extension.as_str()) {
            return fail(format!(
                "{} is an image. Use read_image to view it.",
                params.path
            ));
        }
        if params.pages.is_some() {
            return fail("pages only applies to PDF files.");
        }
        if params.limit == Some(0) {
            return fail("limit must be at least 1.");
        }

        let decoded = match decode_text(&bytes) {
            Ok(decoded) => decoded,
            Err(message) => return fail(format!("{}: {}", params.path, message)),
        };
        ok(render_numbered(
            &decoded.text,
            decoded.note,
            params.line.unwrap_or(1) as usize,
            params.limit.map_or(DEFAULT_LINE_LIMIT, |l| l as usize),
        ))
    }
}

impl Default for ReadTool {
    fn default() -> Self {
        Self::new()
    }
}

struct Decoded {
    text: String,
    note: Option<&'static str>,
}

/// Decodes UTF-8 (with or without BOM) and UTF-16 with BOM; refuses other binary data.
fn decode_text(bytes: &[u8]) -> Result<Decoded, &'static str> {
    let (bom, body) = Bom::detect(bytes);
    match bom {
        Bom::Utf8 => {
            return Ok(Decoded {
                text: bom.decode_lossy(body),
                note: None,
            });
        }
        Bom::Utf16Le => {
            return Ok(Decoded {
                text: bom.decode_lossy(body),
                note: Some(
                    "File is UTF-16 LE (BOM), shown decoded. edit and write keep this encoding.",
                ),
            });
        }
        Bom::Utf16Be => {
            return Ok(Decoded {
                text: bom.decode_lossy(body),
                note: Some(
                    "File is UTF-16 BE (BOM), shown decoded. edit and write keep this encoding.",
                ),
            });
        }
        Bom::None => {}
    }
    if bytes.iter().take(8192).any(|byte| *byte == 0) {
        return Err(
            "this looks like a binary file (or UTF-16 without a BOM), which read cannot display",
        );
    }
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok(Decoded {
            text: text.to_string(),
            note: None,
        }),
        Err(_) => Ok(Decoded {
            text: String::from_utf8_lossy(bytes).into_owned(),
            note: Some("File is not valid UTF-8; invalid bytes are shown as U+FFFD."),
        }),
    }
}

/// `text` cut to at most `max` bytes at a character boundary.
fn cut_to_bytes(text: &str, max: usize) -> String {
    text.chars()
        .scan(0, |len, c| {
            *len += c.len_utf8();
            (*len <= max).then_some(c)
        })
        .collect()
}

fn truncate_line(line: &str) -> String {
    let count = line.chars().count();
    if count <= MAX_LINE_CHARS {
        return line.to_string();
    }
    let kept: String = line.chars().take(MAX_LINE_CHARS).collect();
    format!("{kept}... [line cut: {count} characters]")
}

/// Lines as `{number}\t{text}` (number right-aligned to 6, like `cat -n`), from `start` (1-based),
/// at most `limit` lines and `READ_OUTPUT_LIMIT_BYTES` bytes.
fn render_numbered(text: &str, note: Option<&str>, start: usize, limit: usize) -> String {
    let start = start.max(1);
    let total = text.lines().count();
    if total == 0 {
        return "The file exists but is empty.".to_string();
    }
    if start > total {
        return format!(
            "The file has only {total} lines, which is fewer than the requested start line {start}."
        );
    }

    let mut out = String::new();
    if let Some(note) = note {
        out.push_str(&format!("[{note}]\n"));
    }
    let header_len = out.len();
    let mut last = start - 1;
    let mut byte_capped = false;
    for (index, line) in text.lines().enumerate().skip(start - 1).take(limit) {
        let rendered = format!("{:>6}\t{}\n", index + 1, truncate_line(line));
        if out.len() - header_len + rendered.len() > READ_OUTPUT_LIMIT_BYTES && last >= start {
            byte_capped = true;
            break;
        }
        out.push_str(&rendered);
        last = index + 1;
    }
    if last < total {
        let reason = if byte_capped {
            format!("output limit of {READ_OUTPUT_LIMIT_BYTES} bytes")
        } else {
            format!("limit of {limit} lines")
        };
        out.push_str(&format!(
            "[Showing lines {start}-{last} of {total} ({reason}). Read more with line={}.]",
            last + 1
        ));
    } else if out.ends_with('\n') {
        out.pop();
    }
    out
}

/// Pages from specs like "3", "1-5", "2-" and "1,4-6" (1-based, ascending, no duplicates).
fn parse_pages(spec: &str, total: usize) -> Result<Vec<u32>, String> {
    let mut pages = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        let invalid =
            || format!("Invalid pages value \"{spec}\". Use for example \"3\", \"1-5\" or \"2-\".");
        let (from, to) = match part.split_once('-') {
            Some((from, "")) => (from.trim().parse::<usize>().map_err(|_| invalid())?, total),
            Some((from, to)) => (
                from.trim().parse::<usize>().map_err(|_| invalid())?,
                to.trim().parse::<usize>().map_err(|_| invalid())?,
            ),
            None => {
                let page = part.parse::<usize>().map_err(|_| invalid())?;
                (page, page)
            }
        };
        if from == 0 || to < from {
            return Err(invalid());
        }
        if to > total {
            return Err(format!(
                "The PDF has {total} pages; pages \"{spec}\" goes beyond the last page."
            ));
        }
        for page in from..=to {
            if !pages.contains(&(page as u32)) {
                pages.push(page as u32);
            }
        }
    }
    pages.sort_unstable();
    Ok(pages)
}

fn read_pdf(display: &str, bytes: &[u8], pages: Option<&str>) -> CallToolResult {
    let doc = match lopdf::Document::load_mem(bytes) {
        Ok(doc) => doc,
        Err(error) => {
            return fail(format!(
                "Failed to open PDF {display}: {error}. It may be damaged or password protected."
            ))
        }
    };
    if doc.is_encrypted() {
        return fail(format!(
            "{display} is password protected and cannot be read."
        ));
    }
    let total = doc.get_pages().len();
    if total == 0 {
        return fail(format!("{display} has no pages."));
    }
    let selected: Vec<u32> = match pages {
        Some(spec) => match parse_pages(spec, total) {
            Ok(selected) => selected,
            Err(message) => return fail(message),
        },
        None if total > PDF_PAGES_WITHOUT_RANGE => {
            return fail(format!(
                "{display} has {total} pages. Give pages (at most {MAX_PDF_PAGES_PER_REQUEST} per request), for example pages=\"1-{PDF_PAGES_WITHOUT_RANGE}\"."
            ))
        }
        None => (1..=total as u32).collect(),
    };
    if selected.len() > MAX_PDF_PAGES_PER_REQUEST {
        return fail(format!(
            "At most {MAX_PDF_PAGES_PER_REQUEST} pages per request; {} requested. Split it, for example pages=\"1-{MAX_PDF_PAGES_PER_REQUEST}\".",
            selected.len()
        ));
    }

    let mut out = String::new();
    let mut any_text = false;
    let mut shown_until = 0;
    let mut truncated = false;
    for page in &selected {
        let body = match doc.extract_text_with_limit(&[*page], PDF_PAGE_DECOMPRESS_LIMIT) {
            Ok(text) => {
                any_text |= !text.trim().is_empty();
                text
            }
            Err(error) => format!("(text of this page could not be extracted: {error})"),
        };
        let section = format!("--- Page {page} of {total} ---\n{}\n", body.trim_end());
        if out.len() + section.len() > READ_OUTPUT_LIMIT_BYTES {
            if shown_until == 0 {
                out.push_str(&cut_to_bytes(&section, READ_OUTPUT_LIMIT_BYTES));
                shown_until = *page;
            }
            truncated = true;
            break;
        }
        out.push_str(&section);
        shown_until = *page;
    }
    if truncated {
        out.push_str(&format!(
            "\n[Output limited to {READ_OUTPUT_LIMIT_BYTES} bytes at page {shown_until}. Read the rest with fewer pages per request.]"
        ));
    }
    if !any_text {
        out.push_str(
            "\n[No extractable text found. The PDF may consist of scanned images; read_image cannot open PDF pages.]",
        );
    }
    ok(out)
}

fn notebook_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts.iter().filter_map(Value::as_str).collect(),
        _ => String::new(),
    }
}

fn notebook_output(output: &Value) -> String {
    match output.get("output_type").and_then(Value::as_str) {
        Some("stream") => notebook_text(&output["text"]),
        Some("error") => format!(
            "{}: {}",
            output["ename"].as_str().unwrap_or("Error"),
            output["evalue"].as_str().unwrap_or("")
        ),
        Some("execute_result") | Some("display_data") => {
            let data = &output["data"];
            if data.get("text/plain").is_some() {
                notebook_text(&data["text/plain"])
            } else if data.as_object().is_some_and(|d| !d.is_empty()) {
                "[non-text output omitted]".to_string()
            } else {
                String::new()
            }
        }
        _ => String::new(),
    }
}

fn read_notebook(display: &str, bytes: &[u8]) -> CallToolResult {
    let notebook: Value =
        match serde_json::from_slice(bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes)) {
            Ok(value) => value,
            Err(error) => return fail(format!("{display} is not a valid notebook: {error}")),
        };
    let Some(cells) = notebook.get("cells").and_then(Value::as_array) else {
        return fail(format!("{display} is not a valid notebook: no cells."));
    };
    if cells.is_empty() {
        return ok("The notebook exists but has no cells.");
    }
    let language = notebook["metadata"]["language_info"]["name"]
        .as_str()
        .or_else(|| notebook["metadata"]["kernelspec"]["language"].as_str());

    let mut out = String::new();
    if let Some(language) = language {
        out.push_str(&format!("Notebook language: {language}\n\n"));
    }
    let mut shown = 0;
    let mut truncated = false;
    for (index, cell) in cells.iter().enumerate() {
        let kind = cell["cell_type"].as_str().unwrap_or("code");
        let id = cell["id"]
            .as_str()
            .map(|id| format!(", id {id}"))
            .unwrap_or_default();
        let mut section = format!("--- Cell {} [{kind}{id}] ---\n", index + 1);
        section.push_str(notebook_text(&cell["source"]).trim_end());
        section.push('\n');
        if let Some(outputs) = cell["outputs"].as_array() {
            let text: Vec<String> = outputs
                .iter()
                .map(notebook_output)
                .filter(|text| !text.trim().is_empty())
                .collect();
            if !text.is_empty() {
                section.push_str("Output:\n");
                section.push_str(text.join("\n").trim_end());
                section.push('\n');
            }
        }
        section.push('\n');
        if out.len() + section.len() > READ_OUTPUT_LIMIT_BYTES {
            if shown == 0 {
                out.push_str(&cut_to_bytes(&section, READ_OUTPUT_LIMIT_BYTES - out.len()));
                shown = 1;
            }
            truncated = true;
            break;
        }
        out.push_str(&section);
        shown += 1;
    }
    if truncated {
        out.push_str(&format!(
            "[Output limited to {READ_OUTPUT_LIMIT_BYTES} bytes: showing cells 1-{shown} of {}.]",
            cells.len()
        ));
    }
    ok(out.trim_end().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn text_of(result: &CallToolResult) -> &str {
        match &result.content[0] {
            ContentBlock::Text(text) => &text.text,
            _ => panic!("expected text"),
        }
    }

    fn read(
        dir: &TempDir,
        name: &str,
        bytes: &[u8],
        line: Option<u32>,
        limit: Option<u32>,
    ) -> CallToolResult {
        let path = dir.path().join(name);
        fs::write(&path, bytes).unwrap();
        ReadTool::new().file_read_with_cwd(
            FileReadParams {
                path: path.to_string_lossy().to_string(),
                line,
                limit,
                pages: None,
            },
            None,
        )
    }

    #[test]
    fn numbers_lines_like_cat_n() {
        let dir = tempfile::tempdir().unwrap();
        let result = read(&dir, "a.txt", b"one\ntwo\r\nthree", None, None);
        assert!(!result.is_error.unwrap_or(false));
        assert_eq!(text_of(&result), "     1\tone\n     2\ttwo\n     3\tthree");
    }

    #[test]
    fn partial_read_keeps_real_line_numbers() {
        let dir = tempfile::tempdir().unwrap();
        let result = read(&dir, "a.txt", b"one\ntwo\nthree\nfour", Some(2), Some(2));
        assert_eq!(
            text_of(&result),
            "     2\ttwo\n     3\tthree\n[Showing lines 2-3 of 4 (limit of 2 lines). Read more with line=4.]"
        );
    }

    #[test]
    fn default_limit_is_2000_lines() {
        let dir = tempfile::tempdir().unwrap();
        let content = "x\n".repeat(2500);
        let result = read(&dir, "long.txt", content.as_bytes(), None, None);
        let text = text_of(&result);
        assert!(text.contains("\n  2000\tx\n"));
        assert!(!text.contains("  2001\tx"));
        assert!(text.ends_with(
            "[Showing lines 1-2000 of 2500 (limit of 2000 lines). Read more with line=2001.]"
        ));
    }

    #[test]
    fn byte_cap_stops_before_the_limit() {
        let dir = tempfile::tempdir().unwrap();
        let content = format!("{}\n", "y".repeat(1000)).repeat(500);
        let result = read(&dir, "wide.txt", content.as_bytes(), None, None);
        let text = text_of(&result);
        assert!(text.len() < READ_OUTPUT_LIMIT_BYTES + 200);
        assert!(text.contains("output limit of 100000 bytes"));
        assert!(text.contains("Read more with line="));
    }

    #[test]
    fn long_lines_are_cut_and_marked() {
        let dir = tempfile::tempdir().unwrap();
        let content = "z".repeat(3000);
        let result = read(&dir, "line.txt", content.as_bytes(), None, None);
        assert!(text_of(&result).contains("... [line cut: 3000 characters]"));
    }

    #[test]
    fn empty_file_and_start_beyond_end_are_plain_messages() {
        let dir = tempfile::tempdir().unwrap();
        let empty = read(&dir, "empty.txt", b"", None, None);
        assert!(!empty.is_error.unwrap_or(false));
        assert_eq!(text_of(&empty), "The file exists but is empty.");
        let beyond = read(&dir, "short.txt", b"a\nb", Some(9), None);
        assert!(!beyond.is_error.unwrap_or(false));
        assert!(text_of(&beyond).contains("only 2 lines"));
    }

    #[test]
    fn decodes_bom_and_utf16() {
        let dir = tempfile::tempdir().unwrap();
        let mut utf8_bom = vec![0xEF, 0xBB, 0xBF];
        utf8_bom.extend_from_slice("grün\nzwei".as_bytes());
        let result = read(&dir, "bom.txt", &utf8_bom, None, None);
        assert_eq!(text_of(&result), "     1\tgrün\n     2\tzwei");

        let units: Vec<u16> = "grün\nzwei".encode_utf16().collect();
        let mut le = vec![0xFF, 0xFE];
        le.extend(units.iter().flat_map(|u| u.to_le_bytes()));
        let result = read(&dir, "le.txt", &le, None, None);
        assert!(text_of(&result).contains("UTF-16 LE"));
        assert!(text_of(&result).ends_with("     1\tgrün\n     2\tzwei"));

        let mut be = vec![0xFE, 0xFF];
        be.extend(units.iter().flat_map(|u| u.to_be_bytes()));
        let result = read(&dir, "be.txt", &be, None, None);
        assert!(text_of(&result).contains("UTF-16 BE"));
        assert!(text_of(&result).ends_with("     1\tgrün\n     2\tzwei"));
    }

    #[test]
    fn invalid_utf8_is_shown_with_a_note_and_binary_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let result = read(&dir, "latin.txt", b"caf\xE9\n", None, None);
        assert!(!result.is_error.unwrap_or(false));
        assert!(text_of(&result).contains("not valid UTF-8"));
        assert!(text_of(&result).contains("caf\u{FFFD}"));

        let binary = read(&dir, "data.bin", &[1, 2, 0, 3, 4], None, None);
        assert!(binary.is_error.unwrap_or(false));
        assert!(text_of(&binary).contains("binary file"));
    }

    #[test]
    fn directory_missing_and_image_give_clear_errors() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadTool::new();
        let params = |path: String| FileReadParams {
            path,
            line: None,
            limit: None,
            pages: None,
        };

        let result =
            tool.file_read_with_cwd(params(dir.path().to_string_lossy().to_string()), None);
        assert!(result.is_error.unwrap_or(false));
        assert!(text_of(&result).contains("is a directory"));

        let result = tool.file_read_with_cwd(params("missing.txt".to_string()), Some(dir.path()));
        assert!(result.is_error.unwrap_or(false));
        assert!(text_of(&result).starts_with("File does not exist: missing.txt"));

        let image = read(&dir, "pic.PNG", &[0x89, b'P', b'N', b'G'], None, None);
        assert!(image.is_error.unwrap_or(false));
        assert!(text_of(&image).contains("read_image"));
    }

    #[test]
    fn parses_page_specs() {
        assert_eq!(parse_pages("3", 10).unwrap(), vec![3]);
        assert_eq!(parse_pages("1-3", 10).unwrap(), vec![1, 2, 3]);
        assert_eq!(parse_pages("8-", 10).unwrap(), vec![8, 9, 10]);
        assert_eq!(parse_pages("5,1-2,2", 10).unwrap(), vec![1, 2, 5]);
        assert!(parse_pages("0", 10).is_err());
        assert!(parse_pages("4-2", 10).is_err());
        assert!(parse_pages("x", 10).is_err());
        assert!(parse_pages("9-11", 10).unwrap_err().contains("10 pages"));
    }

    fn pdf_with_pages(texts: &[&str]) -> Vec<u8> {
        use lopdf::content::{Content, Operation};
        use lopdf::{dictionary, Document, Object, Stream};

        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });
        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! { "F1" => font_id },
        });
        let mut kids = Vec::new();
        for text in texts {
            let content = Content {
                operations: vec![
                    Operation::new("BT", vec![]),
                    Operation::new("Tf", vec!["F1".into(), 24.into()]),
                    Operation::new("Td", vec![50.into(), 700.into()]),
                    Operation::new("Tj", vec![Object::string_literal(*text)]),
                    Operation::new("ET", vec![]),
                ],
            };
            let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
            let page_id = doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "Contents" => content_id,
            });
            kids.push(page_id.into());
        }
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => kids,
                "Count" => texts.len() as i64,
                "Resources" => resources_id,
                "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
            }),
        );
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn reads_selected_pdf_pages() {
        let bytes = pdf_with_pages(&["first page", "second page", "third page"]);
        let result = read_pdf("x.pdf", &bytes, Some("2-3"));
        assert!(!result.is_error.unwrap_or(false));
        let text = text_of(&result);
        assert!(text.contains("--- Page 2 of 3 ---"));
        assert!(text.contains("second page"));
        assert!(text.contains("third page"));
        assert!(!text.contains("first page"));

        let all = read_pdf("x.pdf", &bytes, None);
        assert!(text_of(&all).contains("first page"));

        let beyond = read_pdf("x.pdf", &bytes, Some("4"));
        assert!(beyond.is_error.unwrap_or(false));
    }

    #[test]
    fn long_pdf_needs_a_page_range() {
        let texts: Vec<String> = (1..=12).map(|n| format!("page number {n}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let bytes = pdf_with_pages(&refs);

        let result = read_pdf("long.pdf", &bytes, None);
        assert!(result.is_error.unwrap_or(false));
        assert!(text_of(&result).contains("has 12 pages"));

        let result = read_pdf("long.pdf", &bytes, Some("11-12"));
        assert!(!result.is_error.unwrap_or(false));
        assert!(text_of(&result).contains("page number 12"));
    }

    #[test]
    fn invalid_pdf_is_a_clear_error() {
        let result = read_pdf("bad.pdf", b"not a pdf", None);
        assert!(result.is_error.unwrap_or(false));
        assert!(text_of(&result).contains("Failed to open PDF bad.pdf"));
    }

    #[test]
    fn renders_notebook_cells_with_text_outputs() {
        let notebook = r##"{
            "metadata": {"language_info": {"name": "python"}},
            "cells": [
                {"cell_type": "markdown", "id": "a1", "source": ["# Title\n", "text"]},
                {"cell_type": "code", "id": "b2", "source": "print(1)\n1/0",
                 "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": ["1\n"]},
                    {"output_type": "error", "ename": "ZeroDivisionError", "evalue": "division by zero", "traceback": []},
                    {"output_type": "display_data", "data": {"image/png": "AAAA"}}
                 ]}
            ]
        }"##;
        let result = read_notebook("n.ipynb", notebook.as_bytes());
        assert!(!result.is_error.unwrap_or(false));
        let text = text_of(&result);
        assert!(text.starts_with("Notebook language: python"));
        assert!(text.contains("--- Cell 1 [markdown, id a1] ---\n# Title\ntext"));
        assert!(text.contains("--- Cell 2 [code, id b2] ---\nprint(1)\n1/0\nOutput:\n1\n\nZeroDivisionError: division by zero\n[non-text output omitted]"));

        let broken = read_notebook("n.ipynb", b"{");
        assert!(broken.is_error.unwrap_or(false));
        assert!(text_of(&broken).contains("not a valid notebook"));
    }
}
