//! Tool search: defer the definitions of extension tools until the model asks for them,
//! like Claude Code's MCP tool search (C161).
//!
//! With `GOOSE_TOOL_SEARCH` set to true, the tools of every extension that is not part of goose
//! itself (stdio, streamable HTTP and in-process MCP servers) are left out of the tool list sent to
//! the model. The system prompt lists their names under "Deferred tools", and `tool_search` loads
//! the ones a task needs: from the next request on they are in the tool list. A server keeps a tool
//! in the list from the start by setting `_meta["anthropic/alwaysLoad"]` to true, the key Claude
//! Code reads.
//!
//! What has been loaded is read from the conversation, not stored separately: the names a
//! `tool_search` result carries in its `_meta`, plus every tool the conversation has called. The set
//! only grows, so the tool list (the start of every cached prompt) changes only when a tool loads.

use std::collections::HashSet;
use std::sync::Arc;

use rmcp::model::{CallToolResult, ContentBlock, MetaObject, Tool, ToolAnnotations};
use schemars::{schema_for, JsonSchema};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::agents::extension_manager::ExtensionLease;
use crate::config::Config;
use crate::conversation::message::{Message, MessageContent};

pub const TOOL_SEARCH_TOOL_NAME: &str = "tool_search";
/// Config key (environment or config.yaml): true defers extension tools behind `tool_search`.
pub const TOOL_SEARCH_CONFIG_KEY: &str = "GOOSE_TOOL_SEARCH";
/// Per-tool opt-out of deferral, set by the server in the tool's `_meta` (Claude Code's key).
pub const ALWAYS_LOAD_META_KEY: &str = "anthropic/alwaysLoad";
/// Names a `tool_search` result loaded, kept in the result's `_meta` for later requests.
pub const LOADED_TOOLS_META_KEY: &str = "goose_tool_search_loaded";

const DEFAULT_MAX_RESULTS: usize = 5;
const MAX_RESULTS_CAP: usize = 50;
const SUMMARY_CHARS: usize = 200;

pub fn enabled() -> bool {
    Config::global()
        .get_param::<bool>(TOOL_SEARCH_CONFIG_KEY)
        .unwrap_or(false)
}

/// Tool search applies to this lease: switched on (`enabled()`, read once per reply) and not in
/// code mode, where tools are reached through `execute_javascript` instead of the tool list.
pub(crate) fn active_for(switched_on: bool, lease: &ExtensionLease) -> bool {
    #[cfg(feature = "code-mode")]
    if lease.is_enabled(crate::agents::platform_extensions::code_execution::EXTENSION_NAME) {
        return false;
    }
    #[cfg(not(feature = "code-mode"))]
    let _ = lease;
    switched_on
}

pub(crate) fn always_loaded(tool: &Tool) -> bool {
    tool.meta
        .as_ref()
        .and_then(|meta| meta.0.get(ALWAYS_LOAD_META_KEY))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// The tools of a lease, split into the ones the model always gets and the ones it loads on demand.
pub(crate) struct ToolPartition {
    pub upfront: Vec<Tool>,
    pub deferrable: Vec<Tool>,
}

impl ToolPartition {
    /// `entries` pairs each tool with whether its extension is part of goose itself.
    pub fn new(entries: Vec<(Tool, bool)>) -> Self {
        let (upfront, deferrable): (Vec<_>, Vec<_>) = entries
            .into_iter()
            .partition(|(tool, own)| *own || always_loaded(tool));
        Self {
            upfront: upfront.into_iter().map(|(tool, _)| tool).collect(),
            deferrable: deferrable.into_iter().map(|(tool, _)| tool).collect(),
        }
    }

    /// What the model may call now: the upfront tools and the deferred tools already loaded.
    pub fn callable(self, loaded: &HashSet<String>) -> Vec<Tool> {
        let mut tools = self.upfront;
        tools.extend(
            self.deferrable
                .into_iter()
                .filter(|tool| loaded.contains(tool.name.as_ref())),
        );
        tools
    }
}

/// Tools the conversation has loaded through `tool_search` or called.
pub(crate) fn loaded_names(messages: &[Message]) -> HashSet<String> {
    let mut names = HashSet::new();
    for content in messages.iter().flat_map(|message| &message.content) {
        match content {
            MessageContent::ToolRequest(request) => {
                if let Ok(tool_call) = &request.tool_call {
                    names.insert(tool_call.name.to_string());
                }
            }
            MessageContent::ToolResponse(response) => {
                let loaded = response
                    .tool_result
                    .as_ref()
                    .ok()
                    .and_then(|result| result.meta.as_ref())
                    .and_then(|meta| meta.0.get(LOADED_TOOLS_META_KEY))
                    .and_then(Value::as_array);
                names.extend(
                    loaded
                        .into_iter()
                        .flatten()
                        .filter_map(|name| name.as_str().map(str::to_string)),
                );
            }
            _ => {}
        }
    }
    names
}

#[derive(Deserialize, JsonSchema)]
pub(crate) struct ToolSearchParams {
    /// "select:<name>[,<name>...]" loads these exact tools; otherwise keywords, where "+term"
    /// requires the term in the tool name.
    pub(crate) query: String,
    /// How many tools a keyword search loads at most (default 5).
    #[serde(default)]
    pub(crate) max_results: Option<usize>,
}

pub(crate) fn tool() -> Tool {
    let schema = serde_json::to_value(schema_for!(ToolSearchParams))
        .expect("tool_search schema serializes")
        .as_object()
        .cloned()
        .expect("tool_search schema is an object");
    Tool::new(
        TOOL_SEARCH_TOOL_NAME,
        "Loads deferred tools so they can be called.\n\n\
         Deferred tools are listed by name in the system prompt under \"Deferred tools\". Until a \
         tool is loaded only its name is known: there is no parameter schema, and calling it fails \
         as not available. This tool matches a query against the deferred tools and loads the \
         matches; from your next step on they are in your tool list and callable like any other \
         tool.\n\n\
         Query forms:\n\
         - \"select:github__create_issue,github__list_issues\" - load these exact tools by name\n\
         - \"notebook jupyter\" - keyword search, loads up to max_results best matches\n\
         - \"+slack send\" - require \"slack\" in the name, rank by the remaining terms",
        Arc::new(schema),
    )
    .annotate(ToolAnnotations::from_raw(
        Some("Load deferred tools".to_string()),
        Some(true),
        Some(false),
        Some(true),
        Some(false),
    ))
}

/// The system prompt section that names the deferred tools.
pub(crate) fn prompt_section(deferrable: &[Tool]) -> String {
    let mut names: Vec<&str> = deferrable.iter().map(|tool| tool.name.as_ref()).collect();
    names.sort_unstable();
    format!(
        "# Deferred tools\n\n\
         These tools are available on demand; their extensions are already enabled. Until you load \
         them with tool_search only their names are known, and calling one before that fails as not \
         available. Load the ones a task needs \
         with tool_search (query \"select:<name>[,<name>...]\") before calling them:\n{}",
        names.join("\n")
    )
}

pub(crate) struct SearchOutcome<'a> {
    pub matches: Vec<&'a Tool>,
    /// Names asked for with `select:` that match no deferred tool.
    pub unknown: Vec<String>,
}

pub(crate) fn search<'a>(
    query: &str,
    max_results: Option<usize>,
    deferrable: &'a [Tool],
) -> SearchOutcome<'a> {
    let query = query.trim();
    if let Some(names) = strip_prefix_ignore_case(query, "select:") {
        return select(names, deferrable);
    }
    let limit = max_results
        .unwrap_or(DEFAULT_MAX_RESULTS)
        .clamp(1, MAX_RESULTS_CAP);
    let mut required = Vec::new();
    let mut terms = Vec::new();
    for word in query.split_whitespace() {
        let word = word.to_lowercase();
        match word.strip_prefix('+') {
            Some(term) if !term.is_empty() => required.push(term.to_string()),
            Some(_) => {}
            None => terms.push(word),
        }
    }
    let mut scored: Vec<(usize, &Tool)> = deferrable
        .iter()
        .filter_map(|tool| {
            let name = tool.name.to_lowercase();
            if !required.iter().all(|term| name.contains(term.as_str())) {
                return None;
            }
            let description = tool
                .description
                .as_deref()
                .unwrap_or_default()
                .to_lowercase();
            let parts: Vec<&str> = name
                .split(['_', '-'])
                .filter(|part| !part.is_empty())
                .collect();
            let score: usize = terms
                .iter()
                .map(|term| {
                    let in_name = if parts.contains(&term.as_str()) {
                        4
                    } else if name.contains(term.as_str()) {
                        2
                    } else {
                        0
                    };
                    in_name + usize::from(description.contains(term.as_str()))
                })
                .sum();
            // Required terms decide on their own; the other terms only rank.
            (score > 0 || !required.is_empty()).then_some((score, tool))
        })
        .collect();
    scored.sort_by(|(a_score, a), (b_score, b)| b_score.cmp(a_score).then(a.name.cmp(&b.name)));
    SearchOutcome {
        matches: scored
            .into_iter()
            .take(limit)
            .map(|(_, tool)| tool)
            .collect(),
        unknown: Vec::new(),
    }
}

fn strip_prefix_ignore_case<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let head = text.get(..prefix.len())?;
    let rest = text.get(prefix.len()..)?;
    head.eq_ignore_ascii_case(prefix).then_some(rest)
}

/// Exact names first; a name without its extension prefix matches when exactly one tool has it.
fn select<'a>(names: &str, deferrable: &'a [Tool]) -> SearchOutcome<'a> {
    let mut matches: Vec<&Tool> = Vec::new();
    let mut unknown = Vec::new();
    for name in names
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
        let exact = deferrable
            .iter()
            .find(|tool| tool.name == name)
            .or_else(|| {
                deferrable
                    .iter()
                    .find(|tool| tool.name.eq_ignore_ascii_case(name))
            });
        let found = exact.or_else(|| {
            let mut short = deferrable.iter().filter(|tool| {
                tool.name
                    .rsplit_once("__")
                    .is_some_and(|(_, short)| short.eq_ignore_ascii_case(name))
            });
            match (short.next(), short.next()) {
                (Some(tool), None) => Some(tool),
                _ => None,
            }
        });
        match found {
            Some(tool) if !matches.iter().any(|known| known.name == tool.name) => {
                matches.push(tool)
            }
            Some(_) => {}
            None => unknown.push(name.to_string()),
        }
    }
    SearchOutcome { matches, unknown }
}

/// The `tool_search` result: what loaded, in text for the model and in `_meta` for later requests.
pub(crate) fn result(
    query: &str,
    outcome: &SearchOutcome<'_>,
    loaded_before: &HashSet<String>,
) -> CallToolResult {
    let mut lines = Vec::new();
    if outcome.matches.is_empty() {
        lines.push(format!(
            "No deferred tool matches \"{query}\". The deferred tools are listed in the system \
             prompt under Deferred tools; load one by its exact name with \"select:<name>\"."
        ));
    } else {
        lines.push(
            "Loaded; these tools are in your tool list now and can be called directly:".to_string(),
        );
        for tool in &outcome.matches {
            let already = if loaded_before.contains(tool.name.as_ref()) {
                " (already loaded)"
            } else {
                ""
            };
            lines.push(format!(
                "- {}{}: {}",
                tool.name,
                already,
                summary(tool.description.as_deref().unwrap_or_default())
            ));
        }
    }
    if !outcome.unknown.is_empty() {
        lines.push(format!(
            "Not found among the deferred tools: {}.",
            outcome.unknown.join(", ")
        ));
    }
    let mut result = CallToolResult::success(vec![ContentBlock::text(lines.join("\n"))]);
    if !outcome.matches.is_empty() {
        let names: Vec<&str> = outcome
            .matches
            .iter()
            .map(|tool| tool.name.as_ref())
            .collect();
        let mut meta = serde_json::Map::new();
        meta.insert(LOADED_TOOLS_META_KEY.to_string(), json!(names));
        result.meta = Some(MetaObject(meta));
    }
    result
}

fn summary(description: &str) -> String {
    let first = description.trim().lines().next().unwrap_or_default().trim();
    if first.chars().count() <= SUMMARY_CHARS {
        return first.to_string();
    }
    let cut: String = first.chars().take(SUMMARY_CHARS).collect();
    format!("{}…", cut.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::CallToolRequestParams;
    use rmcp::object;

    fn tool(name: &str, description: &str) -> Tool {
        Tool::new(
            name.to_string(),
            description.to_string(),
            Arc::new(serde_json::Map::new()),
        )
    }

    fn always_load(mut tool: Tool, value: bool) -> Tool {
        let mut meta = serde_json::Map::new();
        meta.insert(ALWAYS_LOAD_META_KEY.to_string(), Value::Bool(value));
        tool.meta = Some(MetaObject(meta));
        tool
    }

    fn names(tools: &[&Tool]) -> Vec<String> {
        tools.iter().map(|tool| tool.name.to_string()).collect()
    }

    fn catalog() -> Vec<Tool> {
        vec![
            tool("slack__send_message", "Send a message to a Slack channel"),
            tool("slack__list_channels", "List the channels of the workspace"),
            tool("mail__send-mail", "Send an email"),
            tool("notebook__edit_cell", "Edit a cell of a Jupyter notebook"),
            tool("context7__query-docs", "Query library documentation"),
        ]
    }

    #[test]
    fn own_tools_and_always_loaded_tools_stay_upfront() {
        let partition = ToolPartition::new(vec![
            (tool("shell", "Run a command"), true),
            (tool("tavily__tavily_search", "Search the web"), false),
            (always_load(tool("fragen__ask_user", "Ask"), true), false),
            (always_load(tool("x__y", "Deferred anyway"), false), false),
        ]);
        let upfront: Vec<&str> = partition.upfront.iter().map(|t| t.name.as_ref()).collect();
        let deferrable: Vec<&str> = partition
            .deferrable
            .iter()
            .map(|t| t.name.as_ref())
            .collect();
        assert_eq!(upfront, ["shell", "fragen__ask_user"]);
        assert_eq!(deferrable, ["tavily__tavily_search", "x__y"]);

        let loaded = HashSet::from(["x__y".to_string()]);
        let callable: Vec<String> = partition
            .callable(&loaded)
            .iter()
            .map(|t| t.name.to_string())
            .collect();
        assert_eq!(callable, ["shell", "fragen__ask_user", "x__y"]);
    }

    #[test]
    fn select_takes_exact_names_and_unique_short_names() {
        let catalog = catalog();
        let outcome = search(
            "select:slack__send_message, edit_cell,send,query-docs,missing",
            None,
            &catalog,
        );
        assert_eq!(
            names(&outcome.matches),
            [
                "slack__send_message",
                "notebook__edit_cell",
                "context7__query-docs"
            ]
        );
        assert_eq!(outcome.unknown, ["send", "missing"]);

        let outcome = search(
            "SELECT:Slack__Send_Message,slack__send_message",
            None,
            &catalog,
        );
        assert_eq!(names(&outcome.matches), ["slack__send_message"]);
        assert!(outcome.unknown.is_empty());
    }

    #[test]
    fn keywords_rank_name_parts_over_descriptions_and_respect_the_limit() {
        let catalog = catalog();
        let outcome = search("notebook jupyter", None, &catalog);
        assert_eq!(names(&outcome.matches), ["notebook__edit_cell"]);

        let outcome = search("send", Some(1), &catalog);
        assert_eq!(names(&outcome.matches), ["mail__send-mail"]);

        let outcome = search("send", None, &catalog);
        assert_eq!(
            names(&outcome.matches),
            ["mail__send-mail", "slack__send_message"]
        );

        assert!(search("weather", None, &catalog).matches.is_empty());
    }

    #[test]
    fn required_terms_filter_by_name() {
        let catalog = catalog();
        let outcome = search("+slack send", None, &catalog);
        assert_eq!(
            names(&outcome.matches),
            ["slack__send_message", "slack__list_channels"]
        );
        let outcome = search("+slack", None, &catalog);
        assert_eq!(
            names(&outcome.matches),
            ["slack__list_channels", "slack__send_message"]
        );
        assert!(search("+mail list", None, &catalog).matches.len() == 1);
    }

    #[test]
    fn loaded_names_come_from_results_and_calls() {
        let catalog = catalog();
        let outcome = search("select:slack__send_message", None, &catalog);
        let response = result("select:slack__send_message", &outcome, &HashSet::new());
        let messages = vec![
            Message::user().with_tool_response("search-1", Ok(response)),
            Message::assistant().with_tool_request(
                "call-1",
                Ok(CallToolRequestParams::new("mail__send-mail")
                    .with_arguments(object!({ "to": "a" }))),
            ),
        ];
        let loaded = loaded_names(&messages);
        assert!(loaded.contains("slack__send_message"));
        assert!(loaded.contains("mail__send-mail"));
        assert!(!loaded.contains("slack__list_channels"));
    }

    #[test]
    fn result_names_what_loaded_and_what_was_not_found() {
        let catalog = catalog();
        let outcome = search("select:slack__send_message,nope", None, &catalog);
        let before = HashSet::from(["slack__send_message".to_string()]);
        let response = result("select:slack__send_message,nope", &outcome, &before);
        let text = response.content[0].as_text().unwrap().text.clone();
        assert!(text.contains("- slack__send_message (already loaded): Send a message"));
        assert!(text.contains("Not found among the deferred tools: nope."));

        let empty = search("weather", None, &catalog);
        let response = result("weather", &empty, &HashSet::new());
        assert!(response.meta.is_none());
        assert!(response.content[0]
            .as_text()
            .unwrap()
            .text
            .starts_with("No deferred tool matches \"weather\""));
    }

    #[test]
    fn prompt_section_lists_sorted_names() {
        let section = prompt_section(&catalog());
        assert!(section.starts_with("# Deferred tools"));
        assert!(section.ends_with(
            "context7__query-docs\nmail__send-mail\nnotebook__edit_cell\nslack__list_channels\nslack__send_message"
        ));
    }
}
