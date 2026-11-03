//! Makes filesystem skills available to inference and slash commands.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use goose_sdk_types::custom_requests::{SourceEntry, SourceType};
use rmcp::model::{CallToolResult, ContentBlock, ErrorData, JsonObject, Tool};
use schemars::{schema_for, JsonSchema};
use serde::Deserialize;
use serde_json::Value;
use tracing_futures::Instrument;

use crate::agents::extension_manager::ExtensionLease;
use crate::agents::state_machine::ops_toolcalling::{
    emit_post_tool_use, pending_advertised_tool_requests, run_pre_tool_hooks, tool_span,
    ToolDisposition, EXPIRED_APPROVAL_RESPONSE,
};
use crate::agents::state_machine::{
    applied, messages_since_kickoff, not_applicable, yielded_with, ConversationEffect, Emitter,
    GooseEffect, Operation, OperationResult, SlashCommand,
};
use crate::agents::tool_execution::{declined_response, CHAT_MODE_TOOL_SKIPPED_RESPONSE};
use crate::config::GooseMode;
use crate::conversation::message::Message;
use crate::conversation::Conversation;
use crate::hooks::HookManager;
use crate::session::Session;

const LOAD_SKILL_TOOL_NAME: &str = "load_skill";

pub struct SkillOperation {
    hook_manager: HookManager,
    extension_lease: Arc<StdMutex<Option<Arc<ExtensionLease>>>>,
}

#[derive(Deserialize, JsonSchema)]
struct LoadSkillParams {
    /// Name of the skill to load. Use "skill-name/path" to load a supporting file.
    name: String,
    /// Optional arguments to provide when loading the skill.
    #[serde(default)]
    args: Option<String>,
}

fn skill_tool() -> Result<Tool> {
    let schema = serde_json::to_value(schema_for!(LoadSkillParams))?;
    let schema = schema
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("load_skill schema is not an object"))?;
    Ok(Tool::new(
        LOAD_SKILL_TOOL_NAME,
        "Load a skill's full content into your context so you can follow its instructions.\n\n\
         Skills are listed in your system instructions. When you need to use one, \
         load it first to get the detailed instructions.",
        schema,
    ))
}

/// Set `GOOSE_NO_SKILL_LIST` to leave skills out of the system prompt and the tool list
/// (saves the skill-list input tokens on every turn for runs that do not use skills).
fn skills_disabled() -> bool {
    std::env::var_os("GOOSE_NO_SKILL_LIST").is_some_and(|value| !value.is_empty())
}

/// Skill listing rules of Claude Code (C152), switched on by `GOOSE_SKILL_LISTING` (JSON), which the
/// Goose App server fills from Claude Code's settings for Goose Code. Claude Code (2.x, `aun`/`lun`/`oVt`
/// in its bundle): the listing gets `window * 4 * budgetFraction` characters (default 1 %, window 200,000
/// when unknown), each description at most `maxDescChars` (default 1,536); over budget, names stay and
/// descriptions are dropped starting with the least used skill (score from `skillUsage`); bundled skills
/// and `name-only` overrides are never ranked; `off` and `user-invocable-only` skills are not listed.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ListingRules {
    #[serde(default)]
    overrides: std::collections::HashMap<String, String>,
    #[serde(default)]
    usage: std::collections::HashMap<String, f64>,
    budget_fraction: Option<f64>,
    max_desc_chars: Option<usize>,
}

const CLAUDE_BUDGET_FRACTION: f64 = 0.01;
const CLAUDE_CHARS_PER_TOKEN: f64 = 4.0;
const CLAUDE_FALLBACK_WINDOW: usize = 200_000;
const CLAUDE_MAX_DESC_CHARS: usize = 1536;

fn listing_rules() -> Option<ListingRules> {
    let raw = std::env::var("GOOSE_SKILL_LISTING").ok()?;
    match serde_json::from_str(&raw) {
        Ok(rules) => Some(rules),
        Err(error) => {
            tracing::warn!("GOOSE_SKILL_LISTING is not valid JSON, listing all skills: {error}");
            None
        }
    }
}

/// Context window of the session's model: the size goose learned for this model, else the configured
/// limit, else Claude Code's fallback.
fn context_window(session: &Session) -> usize {
    use crate::session::{ContextWindowState, ExtensionState};
    let model = session.model_config.as_ref().map(|m| m.model_name.as_str());
    ContextWindowState::from_extension_data(&session.extension_data)
        .filter(|state| Some(state.model.as_str()) == model && state.size > 0)
        .map(|state| state.size)
        .or_else(|| session.model_config.as_ref().and_then(|m| m.context_limit))
        .unwrap_or(CLAUDE_FALLBACK_WINDOW)
}

struct ListingEntry<'a> {
    name: &'a str,
    description: &'a str,
    bundled: bool,
    name_only: bool,
}

fn listing_lines(
    entries: &[ListingEntry],
    rules: Option<&ListingRules>,
    window: usize,
) -> Vec<String> {
    let Some(rules) = rules else {
        return entries
            .iter()
            .map(|e| format!("- {}: {}", e.name, e.description))
            .collect();
    };
    let max_desc = rules.max_desc_chars.unwrap_or(CLAUDE_MAX_DESC_CHARS).max(1);
    let fraction = rules.budget_fraction.unwrap_or(CLAUDE_BUDGET_FRACTION);
    let budget = ((window as f64 * CLAUDE_CHARS_PER_TOKEN * fraction).floor() as i64).max(1);
    let name_len = |e: &ListingEntry| e.name.chars().count() as i64;
    let entry_len = |e: &ListingEntry| {
        if e.name_only {
            name_len(e) + 2
        } else {
            name_len(e) + 4 + e.description.chars().count().min(max_desc) as i64
        }
    };
    let separators = entries.len().saturating_sub(1) as i64;
    let total: i64 = entries.iter().map(entry_len).sum::<i64>() + separators;
    let mut without_description = std::collections::HashSet::new();
    if total > budget {
        let fixed = |e: &ListingEntry| e.bundled || e.name_only;
        let base: i64 = entries
            .iter()
            .map(|e| {
                if fixed(e) {
                    entry_len(e)
                } else {
                    name_len(e) + 2
                }
            })
            .sum::<i64>()
            + separators;
        let mut left = budget - base;
        let score = |e: &ListingEntry| rules.usage.get(e.name).copied().unwrap_or(0.0);
        let mut ranked: Vec<&ListingEntry> = entries.iter().filter(|e| !fixed(e)).collect();
        ranked.sort_by(|a, b| score(b).total_cmp(&score(a)));
        for e in ranked {
            let extra = entry_len(e) - (name_len(e) + 2);
            if extra <= left {
                left -= extra;
            } else {
                without_description.insert(e.name);
            }
        }
    }
    entries
        .iter()
        .map(|e| {
            if e.name_only || without_description.contains(e.name) {
                format!("- {}", e.name)
            } else if e.description.chars().count() > max_desc {
                let cut: String = e.description.chars().take(max_desc - 1).collect();
                format!("- {}: {cut}\u{2026}", e.name)
            } else {
                format!("- {}: {}", e.name, e.description)
            }
        })
        .collect()
}

fn skill_instructions(session: &Session) -> Option<String> {
    if skills_disabled() {
        return None;
    }
    let sources = crate::skills::discover_skills(Some(&session.working_dir));
    let mut skills: Vec<&SourceEntry> = sources
        .iter()
        .filter(|source| {
            matches!(
                source.source_type,
                SourceType::Skill | SourceType::BuiltinSkill
            )
        })
        .collect();
    skills.sort_by(|a, b| (&a.name, &a.path).cmp(&(&b.name, &b.path)));
    let rules = listing_rules();
    let mode = |name: &str| {
        rules
            .as_ref()
            .and_then(|r| r.overrides.get(name))
            .map(String::as_str)
            .unwrap_or("on")
    };
    let entries: Vec<ListingEntry> = skills
        .iter()
        .filter(|s| !matches!(mode(&s.name), "off" | "user-invocable-only"))
        .map(|s| ListingEntry {
            name: &s.name,
            description: &s.description,
            bundled: s.source_type == SourceType::BuiltinSkill,
            name_only: mode(&s.name) == "name-only",
        })
        .collect();
    if entries.is_empty() {
        return None;
    }

    let mut instructions = String::from(
        "# Skills\n\nYou have these skills at your disposal. Load one when it can help with the task or when the user asks for it:",
    );
    for line in listing_lines(&entries, rules.as_ref(), context_window(session)) {
        instructions.push('\n');
        instructions.push_str(&line);
    }
    Some(instructions)
}

fn execute_skill(working_dir: &Path, arguments: Option<JsonObject>) -> CallToolResult {
    let params = arguments
        .map(Value::Object)
        .ok_or_else(|| "Missing arguments".to_string())
        .and_then(|arguments| {
            serde_json::from_value::<LoadSkillParams>(arguments)
                .map_err(|error| format!("Invalid arguments: {error}"))
        });
    let params = match params {
        Ok(params) => params,
        Err(error) => return CallToolResult::error(vec![ContentBlock::text(error)]),
    };
    let skill_name = params.name.as_str();
    let args = params.args.as_deref();
    let skills = crate::skills::discover_skills(Some(working_dir));

    if let Some(skill) = skills.iter().find(|skill| skill.name == skill_name) {
        return match crate::skills::loaded_skill_context_with_args(skill, args) {
            Ok(rendered) => CallToolResult::success(vec![ContentBlock::text(rendered)]),
            Err(error) => CallToolResult::error(vec![ContentBlock::text(format!(
                "Failed to parse skill arguments: {error}"
            ))]),
        };
    }

    if let Some((parent_skill_name, raw_relative_path)) = skill_name.split_once('/') {
        let relative_path = raw_relative_path.replace('\\', "/");
        if let Some(skill) = skills.iter().find(|skill| {
            skill.name == parent_skill_name
                && matches!(
                    skill.source_type,
                    SourceType::Skill | SourceType::BuiltinSkill
                )
        }) {
            return load_supporting_file(skill, skill_name, &relative_path);
        }
    }

    let suggestions: Vec<&str> = skills
        .iter()
        .filter(|skill| {
            skill
                .name
                .to_lowercase()
                .contains(&skill_name.to_lowercase())
                || skill_name
                    .to_lowercase()
                    .contains(&skill.name.to_lowercase())
        })
        .take(3)
        .map(|skill| skill.name.as_str())
        .collect();
    if suggestions.is_empty() {
        CallToolResult::error(vec![ContentBlock::text(format!(
            "Skill '{skill_name}' not found."
        ))])
    } else {
        CallToolResult::error(vec![ContentBlock::text(format!(
            "Skill '{skill_name}' not found. Did you mean: {}?",
            suggestions.join(", ")
        ))])
    }
}

fn load_supporting_file(
    skill: &SourceEntry,
    skill_name: &str,
    relative_path: &str,
) -> CallToolResult {
    let skill_dir = PathBuf::from(&skill.path);
    for file_path in &skill.supporting_files {
        let file_path = Path::new(file_path);
        let Ok(relative) = file_path.strip_prefix(&skill_dir) else {
            continue;
        };
        if relative.to_string_lossy().replace('\\', "/") != relative_path {
            continue;
        }
        return match crate::skills::load_supporting_file(&skill_dir, relative, skill_name) {
            Ok(content) => CallToolResult::success(vec![ContentBlock::text(content)]),
            Err(error) => CallToolResult::error(vec![ContentBlock::text(format!(
                "Failed to read '{skill_name}': {error}"
            ))]),
        };
    }

    let available: Vec<String> = skill
        .supporting_files
        .iter()
        .filter_map(|file| {
            Path::new(file)
                .strip_prefix(&skill_dir)
                .ok()
                .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        })
        .take(10)
        .collect();
    if available.is_empty() {
        CallToolResult::error(vec![ContentBlock::text(format!(
            "Skill '{}' has no supporting files.",
            skill.name
        ))])
    } else {
        CallToolResult::error(vec![ContentBlock::text(format!(
            "File '{skill_name}' not found. Available: {}",
            available.join(", ")
        ))])
    }
}

impl SkillOperation {
    pub fn new(
        hook_manager: HookManager,
        extension_lease: Arc<StdMutex<Option<Arc<ExtensionLease>>>>,
    ) -> Self {
        Self {
            hook_manager,
            extension_lease,
        }
    }

    fn leased_session(&self, session: &Session) -> Option<Session> {
        let lease = self
            .extension_lease
            .lock()
            .expect("extension lease unavailable")
            .clone()?;
        if lease.scope_id() != session.id.as_str() {
            return None;
        }
        let mut session = session.clone();
        session.working_dir = lease.working_dir()?.to_path_buf();
        Some(session)
    }

    fn command_response(
        conversation: &Conversation,
        message: String,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let command = messages_since_kickoff(conversation)?
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("skill command conversation has no kickoff message"))?;
        let message_id = command
            .id
            .clone()
            .ok_or_else(|| anyhow!("Persisted slash command message has no id"))?;
        let command = command.with_visibility(true, false);
        let response = Message::assistant()
            .with_text(message)
            .with_visibility(true, false);
        emit.message(command);
        let response = emit.message(response);
        yielded_with([
            ConversationEffect::SetMessageVisibility {
                message_id,
                user_visible: true,
                agent_visible: false,
            }
            .into(),
            response.into(),
        ])
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for SkillOperation {
    fn name(&self) -> &'static str {
        "skills"
    }

    async fn run_command(
        &self,
        command: &SlashCommand<'_>,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        if command.command == "skills" {
            return Self::command_response(
                conversation,
                crate::slash_commands::skill_slash_command::format_installed_skills(Some(
                    &session.working_dir,
                )),
                emit,
            );
        }

        let prompt = match crate::slash_commands::skill_slash_command::resolve_command(
            command.command,
            command.params_str,
            Some(&session.working_dir),
        ) {
            Ok(Some(prompt)) => prompt,
            Ok(None) => return not_applicable(),
            Err(error) => return Self::command_response(conversation, error, emit),
        };
        let command_message = messages_since_kickoff(conversation)?
            .first()
            .ok_or_else(|| anyhow!("skill command conversation has no kickoff message"))?;
        let message_id = command_message
            .id
            .clone()
            .ok_or_else(|| anyhow!("Persisted slash command message has no id"))?;
        applied([
            ConversationEffect::SetMessageVisibility {
                message_id,
                user_visible: true,
                agent_visible: false,
            }
            .into(),
            Message::user()
                .with_text(prompt)
                .with_visibility(false, true)
                .into(),
        ])
    }

    async fn inference_tools(&self, _session: &Session) -> Result<Vec<Tool>> {
        if skills_disabled() {
            return Ok(Vec::new());
        }
        Ok(vec![skill_tool()?])
    }

    async fn prompt_parts(
        &self,
        session: &Session,
        _conversation: &Conversation,
    ) -> Result<Vec<(String, String)>> {
        Ok(skill_instructions(session)
            .map(|instructions| ("skills".to_string(), instructions))
            .into_iter()
            .collect())
    }

    async fn run(
        &self,
        session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let messages = messages_since_kickoff(conversation)?;
        let pending: Vec<_> = pending_advertised_tool_requests(messages)
            .into_iter()
            .filter(|(request, _)| {
                request
                    .tool_call
                    .as_ref()
                    .is_ok_and(|tool_call| tool_call.name.as_ref() == LOAD_SKILL_TOOL_NAME)
            })
            .collect();
        if pending.is_empty() {
            return not_applicable();
        }

        let leased_session = self.leased_session(session);
        let mut response = Message::user();
        for (request, disposition) in pending {
            let result: std::result::Result<CallToolResult, ErrorData> = match disposition {
                ToolDisposition::Execute if session.goose_mode == GooseMode::Chat => {
                    // Nothing executes in chat mode, so no tool lifecycle runs.
                    Ok(CallToolResult::success(vec![ContentBlock::text(
                        CHAT_MODE_TOOL_SKIPPED_RESPONSE,
                    )]))
                }
                ToolDisposition::Execute => {
                    let Some(session) = leased_session.as_ref() else {
                        response.add_tool_response_with_metadata(
                            request.id,
                            Ok(CallToolResult::error(vec![ContentBlock::text(
                                EXPIRED_APPROVAL_RESPONSE,
                            )])),
                            request.metadata.as_ref(),
                        );
                        continue;
                    };
                    let tool_call = request.tool_call.as_ref().map_err(|error| {
                        anyhow!("load_skill tool call could not be parsed: {error}")
                    })?;
                    let span = tool_span(&tool_call.name, &request.id, &session.id);
                    // `load_skill` is executed here rather than by
                    // ToolExecutionOperation, which is registered after this one.
                    // Run the same hook lifecycle it would have run there.
                    let tool_input = tool_call
                        .arguments
                        .as_ref()
                        .map(|arguments| Value::Object(arguments.clone()));
                    match run_pre_tool_hooks(
                        &self.hook_manager,
                        session,
                        &request.id,
                        &tool_call.name,
                        tool_input.as_ref(),
                    )
                    .instrument(span.clone())
                    .await
                    {
                        // A denial returns before execution and emits no post
                        // event, the same shape ToolExecutionOperation has: its
                        // dispatch returns the denial before the post-hook wrapper
                        // is ever applied.
                        Err(denial) => Err(denial),
                        Ok(()) => {
                            let result = {
                                let _entered = span.enter();
                                execute_skill(&session.working_dir, tool_call.arguments.clone())
                            };
                            if result.is_error == Some(true) {
                                span.record("error.type", "tool_error");
                            }
                            let output = Ok(result);
                            // Post event carries the same tool_call_id as the pre
                            // events. The large-response rewrite
                            // ToolExecutionOperation applies is deliberately not
                            // reused: a skill body is content the model is meant to
                            // read, not a payload to offload to a temp file.
                            emit_post_tool_use(
                                &self.hook_manager,
                                &session.id,
                                &session.working_dir.to_string_lossy(),
                                &tool_call.name,
                                &request.id,
                                tool_input.as_ref(),
                                &output,
                            )
                            .instrument(span.clone())
                            .await;
                            output
                        }
                    }
                }
                ToolDisposition::Decline => Ok(CallToolResult::error(vec![ContentBlock::text(
                    declined_response(&request),
                )])),
                ToolDisposition::ParseError(error) => {
                    Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                        "The tool call could not be parsed: {error}. Correct the arguments and try again."
                    ))]))
                }
            };
            response.add_tool_response_with_metadata(request.id, result, request.metadata.as_ref());
        }
        let response = emit.message(response);
        applied([response.into()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn supporting_file_loader_reads_nested_regular_file() {
        let root = tempfile::tempdir().unwrap();
        let skill_dir = std::fs::canonicalize(root.path()).unwrap();
        let nested = skill_dir.join("nested");
        std::fs::create_dir(&nested).unwrap();
        let file = nested.join("guide.md");
        std::fs::write(&file, "Nested guidance.").unwrap();
        let skill = SourceEntry {
            source_type: SourceType::Skill,
            name: "test-skill".to_string(),
            description: String::new(),
            content: String::new(),
            path: skill_dir.to_string_lossy().into_owned(),
            global: false,
            writable: true,
            supporting_files: vec![file.to_string_lossy().into_owned()],
            properties: HashMap::new(),
        };

        let result = load_supporting_file(&skill, "test-skill/nested/guide.md", "nested/guide.md");

        assert_eq!(result.is_error, Some(false));
        let text = result.content[0].as_text().expect("expected text");
        assert!(text.text.contains("Nested guidance."));
    }
}
