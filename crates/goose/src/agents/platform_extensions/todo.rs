use crate::agents::extension::PlatformExtensionContext;
use crate::agents::mcp_client::{Error, McpClientTrait};
use crate::agents::tool_execution::ToolCallContext;
use crate::session::extension_data;
use crate::session::extension_data::{ExtensionState, TodoItem, TodoStatus};
use anyhow::Result;
use async_trait::async_trait;
use indoc::indoc;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, InitializeResult, JsonObject, ListToolsResult,
    ServerCapabilities, Tool, ToolAnnotations,
};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

pub static EXTENSION_NAME: &str = "todo";
pub const TODO_WRITE_TOOL_NAME: &str = "todo_write";
pub const TODO_WRITE_TOOL_NAME_COMPLETE: &str = "todo__todo_write";

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct TodoWriteParams {
    /// The complete task list; it replaces the previous one. An empty list clears it.
    todos: Vec<TodoItem>,
}

pub struct TodoClient {
    info: InitializeResult,
    context: PlatformExtensionContext,
}

impl TodoClient {
    pub fn new(context: PlatformExtensionContext) -> Result<Self> {
        let info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new(EXTENSION_NAME.to_string(), "1.0.0".to_string())
                    .with_title("Todo"),
            )
            .with_instructions(
                indoc! {r#"
                Your task list is automatically available in your context.

                Use it for work with three or more steps, or when several things are asked at once. Skip it for a single simple step.
                - Each task has a status: pending, in_progress or completed.
                - Mark a task in_progress before you start it, with one task in progress at a time.
                - Mark it completed right when it is fully done, not in batches. Failing tests or partial work stay in_progress; add a task for any blocker you find.
                - Remove tasks that no longer apply.
                - Never redo or re-verify completed work because of this list
            "#}
                .to_string(),
            );

        Ok(Self { info, context })
    }

    async fn handle_write_todo(
        &self,
        session_id: &str,
        arguments: Option<JsonObject>,
    ) -> Result<Vec<ContentBlock>, String> {
        let params: TodoWriteParams = serde_json::from_value(serde_json::Value::Object(
            arguments.ok_or("Missing arguments")?,
        ))
        .map_err(|e| format!("Invalid arguments: {}", e))?;
        let items = params.todos;

        if let Some(position) = items.iter().position(|item| item.content.trim().is_empty()) {
            return Err(format!("Task {} has no content", position + 1));
        }

        let char_count: usize = items
            .iter()
            .map(|item| {
                item.content.chars().count()
                    + item
                        .active_form
                        .as_ref()
                        .map_or(0, |form| form.chars().count())
            })
            .sum();
        let max_chars = std::env::var("GOOSE_TODO_MAX_CHARS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50_000);

        if max_chars > 0 && char_count > max_chars {
            return Err(format!(
                "Todo list too large: {} chars (max: {})",
                char_count, max_chars
            ));
        }

        let done = count_status(&items, TodoStatus::Completed);
        let mut summary = format!("Updated: {} of {} tasks completed", done, items.len());
        if count_status(&items, TodoStatus::InProgress) > 1 {
            summary.push_str(". Keep only one task in_progress at a time");
        }

        self.context
            .session_manager
            .set_extension_state(session_id, &extension_data::TodoState::new(items))
            .await
            .map(|()| vec![ContentBlock::text(summary)])
            .map_err(|_| "Failed to update session metadata".to_string())
    }

    fn get_tools() -> Vec<Tool> {
        let schema = schema_for!(TodoWriteParams);
        let schema_value =
            serde_json::to_value(schema).expect("Failed to serialize TodoWriteParams schema");

        vec![Tool::new(
            TODO_WRITE_TOOL_NAME.to_string(),
            indoc! {r#"
                    Write the task list for this conversation, one entry per task with a status.

                    The list persists across conversation turns and compaction. Use it to plan and
                    track a multi-step task: mark the task you are working on in_progress and each
                    finished task completed.

                    WARNING: This operation completely replaces the existing list. Always send
                    every task you want to keep, not just the changes.
                "#}
            .to_string(),
            schema_value.as_object().unwrap().clone(),
        )
        .annotate(ToolAnnotations::from_raw(
            Some("Write TODO".to_string()),
            Some(false),
            Some(true),
            Some(false),
            Some(false),
        ))]
    }
}

fn count_status(items: &[TodoItem], status: TodoStatus) -> usize {
    items.iter().filter(|item| item.status == status).count()
}

/// The list as the model sees it every turn: one line per task with its status.
fn render_items(items: &[TodoItem]) -> String {
    items
        .iter()
        .map(|item| {
            let mark = match item.status {
                TodoStatus::Pending => "[ ]",
                TodoStatus::InProgress => "[in progress]",
                TodoStatus::Completed => "[x]",
            };
            format!("- {} {}", mark, item.content)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[async_trait]
impl McpClientTrait for TodoClient {
    async fn list_tools(
        &self,
        _session_id: &str,
        _next_cursor: Option<String>,
        _cancellation_token: CancellationToken,
    ) -> Result<ListToolsResult, Error> {
        Ok(ListToolsResult {
            tools: Self::get_tools(),
            next_cursor: None,
            meta: None,
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        ctx: &ToolCallContext,
        name: &str,
        arguments: Option<JsonObject>,
        _cancellation_token: CancellationToken,
    ) -> Result<CallToolResult, Error> {
        let session_id = &ctx.session_id;
        let content = match name {
            TODO_WRITE_TOOL_NAME => self.handle_write_todo(session_id, arguments).await,
            _ => Err(format!("Unknown tool: {}", name)),
        };

        match content {
            Ok(content) => Ok(CallToolResult::success(content)),
            Err(error) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error: {}",
                error
            ))])),
        }
    }

    fn get_info(&self) -> Option<&InitializeResult> {
        Some(&self.info)
    }

    async fn get_moim(&self, session_id: &str, _tools: &[Tool]) -> Option<String> {
        let metadata = self
            .context
            .session_manager
            .get_session(session_id, false)
            .await
            .ok()?;

        match extension_data::TodoState::from_extension_data(&metadata.extension_data) {
            Some(state) if !state.items.is_empty() => Some(format!(
                "Current tasks (keep the statuses up to date):\n{}\n",
                render_items(&state.items)
            )),
            // Sessions from before the list had single tasks
            Some(state) if !state.content.trim().is_empty() => Some(format!(
                "Planning notes (for your reference; items need not be closed out):\n{}\n",
                state.content
            )),
            _ => Some("Current tasks and notes:\n(none)\n".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(content: &str, status: TodoStatus) -> TodoItem {
        TodoItem {
            content: content.to_string(),
            status,
            active_form: None,
        }
    }

    #[test]
    fn list_shows_each_task_with_its_status() {
        let items = vec![
            item("Read the code", TodoStatus::Completed),
            item("Write the fix", TodoStatus::InProgress),
            item("Run the tests", TodoStatus::Pending),
        ];
        assert_eq!(
            render_items(&items),
            "- [x] Read the code\n- [in progress] Write the fix\n- [ ] Run the tests"
        );
    }

    #[test]
    fn schema_asks_for_a_list_of_tasks_with_status() {
        let tool = TodoClient::get_tools().remove(0);
        let schema = serde_json::Value::Object((*tool.input_schema).clone()).to_string();
        for needle in ["todos", "status", "in_progress", "completed", "activeForm"] {
            assert!(schema.contains(needle), "schema lacks {needle}: {schema}");
        }
    }

    #[test]
    fn arguments_parse_like_claude_codes_todo_write() {
        let params: TodoWriteParams = serde_json::from_value(serde_json::json!({
            "todos": [
                {"content": "Run the tests", "status": "in_progress", "activeForm": "Running the tests"},
                {"content": "Ship it", "status": "pending"}
            ]
        }))
        .unwrap();
        assert_eq!(params.todos[0].status, TodoStatus::InProgress);
        assert_eq!(
            params.todos[0].active_form.as_deref(),
            Some("Running the tests")
        );
        assert_eq!(params.todos[1].active_form, None);
    }
}
