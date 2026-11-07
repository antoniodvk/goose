//! What the Codex backend reports when a response ends (C214).
//!
//! Goose decides the end of a turn itself and reads neither `end_turn` nor the
//! `phase` of a text answer, so neither was ever visible. This module only
//! watches the raw stream next to the parser: it changes nothing the model or
//! the agent sees, and a failed write never reaches the chat.
//!
//! Two things are kept per profile under `state/logs`:
//! - `codex_response_ends.jsonl`: one line per response, without any content.
//! - `codex_decision_points/`: the request behind a stop. `llm_request.*.jsonl`
//!   keeps ten requests per profile, which a busy session overwrites in about a
//!   minute, so a stop would be gone before anyone can replay it.

use crate::config::paths::Paths;
use chrono::Utc;
use serde::Serialize;
use serde_json::{json, Value};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const RESPONSE_ENDS_FILE: &str = "logs/codex_response_ends.jsonl";
const DECISION_POINTS_DIR: &str = "logs/codex_decision_points";
const MAX_RESPONSE_ENDS_BYTES: u64 = 5 * 1024 * 1024;
const DECISION_POINTS_TO_KEEP: usize = 100;

pub fn response_ends_path() -> PathBuf {
    Paths::in_state_dir(RESPONSE_ENDS_FILE)
}

pub fn decision_points_dir() -> PathBuf {
    Paths::in_state_dir(DECISION_POINTS_DIR)
}

/// `arguments` can hold file contents, so only the replay tool reads it; the
/// line written per response carries the name alone.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ToolCallSeen {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ResponseEnd {
    pub response_id: Option<String>,
    pub status: Option<String>,
    pub incomplete_reason: Option<String>,
    pub end_turn: Option<bool>,
    /// One entry per assistant message of the response.
    pub phases: Vec<Option<String>>,
    pub tool_calls: Vec<ToolCallSeen>,
    pub reasoning_items: usize,
    /// Field names of the final `response` object, to spot a signal nobody
    /// knows about yet. Names only, no values.
    pub response_fields: Vec<String>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
}

impl ResponseEnd {
    pub fn observe(&mut self, line: &str) {
        let Some(data) = line.strip_prefix("data:") else {
            return;
        };
        let may_matter = [
            "output_item.done",
            "response.completed",
            "response.incomplete",
        ]
        .iter()
        .any(|event| data.contains(event));
        if !may_matter {
            return;
        }
        let Ok(event) = serde_json::from_str::<Value>(data.trim_start()) else {
            return;
        };
        match event.get("type").and_then(Value::as_str) {
            Some("response.output_item.done") => {
                if let Some(item) = event.get("item") {
                    self.note_item(item);
                }
            }
            Some("response.completed" | "response.incomplete") => {
                if let Some(response) = event.get("response") {
                    self.note_final_response(response);
                }
            }
            _ => {}
        }
    }

    fn note_item(&mut self, item: &Value) {
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => self.reasoning_items += 1,
            Some("message") => self
                .phases
                .push(item.get("phase").and_then(Value::as_str).map(String::from)),
            Some("function_call") => self.tool_calls.push(ToolCallSeen {
                name: text_field(item, "name").unwrap_or_default(),
                arguments: text_field(item, "arguments").unwrap_or_default(),
            }),
            _ => {}
        }
    }

    fn note_final_response(&mut self, response: &Value) {
        self.response_id = text_field(response, "id");
        self.status = text_field(response, "status");
        self.incomplete_reason = response
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
            .map(String::from);
        self.end_turn = response.get("end_turn").and_then(Value::as_bool);
        self.input_tokens = response
            .pointer("/usage/input_tokens")
            .and_then(Value::as_i64);
        self.output_tokens = response
            .pointer("/usage/output_tokens")
            .and_then(Value::as_i64);
        if let Some(fields) = response.as_object() {
            self.response_fields = fields.keys().cloned().collect();
        }
        // Like the stream parser: the complete output replaces the items seen so far.
        if let Some(output) = response.get("output").and_then(Value::as_array) {
            if !output.is_empty() {
                self.phases.clear();
                self.tool_calls.clear();
                self.reasoning_items = 0;
                output.iter().for_each(|item| self.note_item(item));
            }
        }
    }

    pub fn is_finished(&self) -> bool {
        self.status.is_some()
    }

    pub fn calls_tool(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// One line without content: tool names, no arguments, no text.
    pub fn summary(&self, session_id: Option<&str>, model: &str, tools_offered: bool) -> Value {
        json!({
            "ts": Utc::now().to_rfc3339(),
            "session": session_id,
            "model": model,
            "response_id": self.response_id,
            "outcome": if self.calls_tool() { "tool_call" } else { "final" },
            "status": self.status,
            "incomplete_reason": self.incomplete_reason,
            "end_turn": self.end_turn,
            "phases": self.phases,
            "tools": self.tool_calls.iter().map(|call| &call.name).collect::<Vec<_>>(),
            "reasoning_items": self.reasoning_items,
            "response_fields": self.response_fields,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tools_offered": tools_offered,
        })
    }
}

fn text_field(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(String::from)
}

/// A request without tools is a title or a summary, never a working step.
pub fn offers_tools(request: &Value) -> bool {
    request
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
}

/// Never fails: the trace must not be able to disturb a chat.
pub fn record_response_end(
    session_id: Option<&str>,
    model: &str,
    request: &Value,
    end: &ResponseEnd,
) {
    let summary = end.summary(session_id, model, offers_tools(request));
    if let Err(error) = append_summary(&response_ends_path(), &summary) {
        tracing::warn!("Codex response end not recorded: {error}");
    }
    if let Err(error) = keep_decision_point(
        &decision_points_dir(),
        session_id,
        request,
        end,
        &summary,
        DECISION_POINTS_TO_KEEP,
    ) {
        tracing::warn!("Codex decision point not kept: {error}");
    }
}

fn append_summary(path: &Path, summary: &Value) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if fs::metadata(path).is_ok_and(|file| file.len() >= MAX_RESPONSE_ENDS_BYTES) {
        fs::rename(path, path.with_extension("old.jsonl"))?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(format!("{summary}\n").as_bytes())
}

fn file_part(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// A text answer ends the turn, so every one is kept: the stop hook can continue
/// the session afterwards. Of the tool calls only the session's latest stays,
/// because its last call is a stop when it hands the session over.
fn keep_decision_point(
    dir: &Path,
    session_id: Option<&str>,
    request: &Value,
    end: &ResponseEnd,
    summary: &Value,
    to_keep: usize,
) -> io::Result<()> {
    if !offers_tools(request) || !end.is_finished() {
        return Ok(());
    }
    fs::create_dir_all(dir)?;
    let session = file_part(session_id.unwrap_or("no-session"));
    let name = if end.calls_tool() {
        format!("{session}.latest.json")
    } else {
        let response_id = end.response_id.as_deref().unwrap_or("no-response-id");
        format!("{session}.{}.json", file_part(response_id))
    };
    let point = json!({ "end": summary, "request": request });
    fs::write(dir.join(name), serde_json::to_vec(&point)?)?;
    prune_oldest(dir, to_keep)
}

fn prune_oldest(dir: &Path, to_keep: usize) -> io::Result<()> {
    let mut files: Vec<_> = fs::read_dir(dir)?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
        .collect();
    if files.len() <= to_keep {
        return Ok(());
    }
    files.sort();
    for (_, path) in &files[..files.len() - to_keep] {
        fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observed(lines: &[String]) -> ResponseEnd {
        let mut end = ResponseEnd::default();
        lines.iter().for_each(|line| end.observe(line));
        end
    }

    fn data(event: Value) -> String {
        format!("data: {event}")
    }

    fn completed(extra: Value) -> String {
        let mut response = json!({
            "id": "resp_1",
            "status": "completed",
            "usage": { "input_tokens": 10, "output_tokens": 4 },
        });
        response
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        data(json!({ "type": "response.completed", "response": response }))
    }

    fn message_done(phase: &str) -> String {
        data(json!({
            "type": "response.output_item.done",
            "item": { "type": "message", "role": "assistant", "phase": phase, "content": [] },
        }))
    }

    fn tool_done() -> String {
        data(json!({
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "developer__shell",
                "arguments": "{\"command\":\"secret\"}",
            },
        }))
    }

    fn tools_request() -> Value {
        json!({ "model": "m", "tools": [{ "name": "developer__shell" }] })
    }

    fn finished(tool_call: bool) -> ResponseEnd {
        let mut lines = vec![completed(json!({}))];
        if tool_call {
            lines.insert(0, tool_done());
        }
        observed(&lines)
    }

    #[test]
    fn a_text_answer_shows_phase_end_turn_and_status() {
        let end = observed(&[
            message_done("final_answer"),
            completed(json!({ "end_turn": false })),
        ]);

        assert_eq!(end.phases, vec![Some("final_answer".to_string())]);
        assert_eq!(end.end_turn, Some(false));
        assert_eq!(end.status.as_deref(), Some("completed"));
        assert!(!end.calls_tool());
    }

    #[test]
    fn a_missing_end_turn_stays_unknown() {
        let end = observed(&[completed(json!({}))]);

        assert_eq!(end.end_turn, None);
    }

    #[test]
    fn the_incomplete_reason_is_kept() {
        let end = observed(&[data(json!({
            "type": "response.incomplete",
            "response": {
                "id": "resp_1",
                "status": "incomplete",
                "incomplete_details": { "reason": "max_output_tokens" },
            },
        }))]);

        assert_eq!(end.status.as_deref(), Some("incomplete"));
        assert_eq!(end.incomplete_reason.as_deref(), Some("max_output_tokens"));
    }

    #[test]
    fn the_complete_output_replaces_the_items_seen_so_far() {
        let end = observed(&[
            message_done("commentary"),
            completed(json!({
                "output": [
                    { "type": "reasoning", "summary": [] },
                    { "type": "message", "phase": "final_answer", "content": [] },
                ],
            })),
        ]);

        assert_eq!(end.phases, vec![Some("final_answer".to_string())]);
        assert_eq!(end.reasoning_items, 1);
    }

    #[test]
    fn text_deltas_and_foreign_lines_change_nothing() {
        let end = observed(&[
            "event: response.completed".to_string(),
            data(json!({
                "type": "response.output_text.delta",
                "delta": "response.completed",
            })),
            "data: not json with response.completed".to_string(),
        ]);

        assert_eq!(end, ResponseEnd::default());
    }

    #[test]
    fn the_line_per_response_holds_no_tool_arguments() {
        let end = finished(true);

        let summary = end.summary(Some("s1"), "gpt-test", true);

        assert_eq!(summary["outcome"], "tool_call");
        assert_eq!(summary["tools"], json!(["developer__shell"]));
        assert!(!summary.to_string().contains("secret"));
    }

    #[test]
    fn response_fields_list_names_without_values() {
        let end = observed(&[completed(json!({ "unknown_signal": "value" }))]);

        assert!(end.response_fields.contains(&"unknown_signal".to_string()));
        assert!(!end.summary(None, "m", true).to_string().contains("value"));
    }

    #[test]
    fn response_ends_are_appended_line_by_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("logs").join("ends.jsonl");

        append_summary(&path, &json!({ "n": 1 })).unwrap();
        append_summary(&path, &json!({ "n": 2 })).unwrap();

        assert_eq!(fs::read_to_string(path).unwrap(), "{\"n\":1}\n{\"n\":2}\n");
    }

    #[test]
    fn a_full_response_end_file_moves_aside() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ends.jsonl");
        let full = vec![b'x'; MAX_RESPONSE_ENDS_BYTES as usize];
        fs::write(&path, full).unwrap();

        append_summary(&path, &json!({ "n": 1 })).unwrap();

        assert_eq!(fs::read_to_string(&path).unwrap(), "{\"n\":1}\n");
        assert!(path.with_extension("old.jsonl").exists());
    }

    #[test]
    fn every_text_answer_stays_as_its_own_decision_point() {
        let dir = tempfile::tempdir().unwrap();
        let end = finished(false);
        let summary = end.summary(Some("s1"), "m", true);

        keep_decision_point(dir.path(), Some("s1"), &tools_request(), &end, &summary, 10).unwrap();

        let point: Value =
            serde_json::from_slice(&fs::read(dir.path().join("s1.resp_1.json")).unwrap()).unwrap();
        assert_eq!(point["request"], tools_request());
        assert_eq!(point["end"]["outcome"], "final");
    }

    #[test]
    fn only_the_latest_tool_call_of_a_session_stays() {
        let dir = tempfile::tempdir().unwrap();
        let end = finished(true);
        let summary = end.summary(Some("s1"), "m", true);

        for _ in 0..2 {
            keep_decision_point(dir.path(), Some("s1"), &tools_request(), &end, &summary, 10)
                .unwrap();
        }

        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
        assert!(dir.path().join("s1.latest.json").exists());
    }

    #[test]
    fn a_request_without_tools_is_no_decision_point() {
        let dir = tempfile::tempdir().unwrap();
        let end = finished(false);
        let summary = end.summary(Some("s1"), "m", false);

        keep_decision_point(dir.path(), Some("s1"), &json!({}), &end, &summary, 10).unwrap();

        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn an_unfinished_response_is_no_decision_point() {
        let dir = tempfile::tempdir().unwrap();
        let end = ResponseEnd::default();
        let summary = end.summary(Some("s1"), "m", true);

        keep_decision_point(dir.path(), Some("s1"), &tools_request(), &end, &summary, 10).unwrap();

        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn session_names_cannot_leave_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        let end = finished(true);
        let summary = end.summary(None, "m", true);

        keep_decision_point(
            dir.path(),
            Some("../x"),
            &tools_request(),
            &end,
            &summary,
            10,
        )
        .unwrap();

        assert!(dir.path().join("___x.latest.json").exists());
    }

    #[test]
    fn the_oldest_decision_points_go_first() {
        let dir = tempfile::tempdir().unwrap();
        for (name, age) in [("old", 300), ("middle", 200), ("new", 100)] {
            let path = dir.path().join(format!("{name}.json"));
            fs::write(&path, "{}").unwrap();
            let modified = std::time::SystemTime::now() - std::time::Duration::from_secs(age);
            fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(modified)
                .unwrap();
        }

        prune_oldest(dir.path(), 2).unwrap();

        assert!(!dir.path().join("old.json").exists());
        assert!(dir.path().join("middle.json").exists());
        assert!(dir.path().join("new.json").exists());
    }
}
