//! Offers `tool_search` and answers its calls when extension tools are deferred (C161).

use std::sync::{Arc, Mutex as StdMutex};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use rmcp::model::{CallToolResult, ContentBlock, ErrorData, Tool};
use serde_json::Value;
use tracing_futures::Instrument;

use crate::agents::state_machine::ops_toolcalling::{
    pending_advertised_tool_requests, tool_span, ToolDisposition,
};
use crate::agents::state_machine::{
    applied, messages_since_kickoff, not_applicable, Emitter, GooseEffect, Operation,
    OperationResult,
};
use crate::agents::tool_search::{self, ToolPartition, ToolSearchParams, TOOL_SEARCH_TOOL_NAME};
use crate::agents::tool_snapshot::ToolSnapshot;
use crate::agents::StateMachineServices;
use crate::conversation::message::Message;
use crate::conversation::Conversation;
use crate::session::Session;

pub struct ToolSearchOperation {
    services: Arc<StateMachineServices>,
    tools: Arc<StdMutex<Option<Arc<ToolSnapshot>>>>,
    switched_on: bool,
}

impl ToolSearchOperation {
    pub fn new(
        services: Arc<StateMachineServices>,
        tools: Arc<StdMutex<Option<Arc<ToolSnapshot>>>>,
        switched_on: bool,
    ) -> Self {
        Self {
            services,
            tools,
            switched_on,
        }
    }

    /// The deferrable tools of the session's snapshot, `None` while tool search does not apply.
    async fn deferrable(&self, session: &Session) -> Result<Option<Vec<Tool>>> {
        if !self.switched_on {
            return Ok(None);
        }
        let snapshot = ToolSnapshot::for_session(&self.tools, &self.services, &session.id).await?;
        if !tool_search::active_for(self.switched_on, &snapshot) {
            return Ok(None);
        }
        let entries: Vec<_> = snapshot
            .mcp
            .tools()
            .await
            .into_iter()
            .map(|tool| (tool, false))
            .collect();
        let partition = ToolPartition::new(entries);
        Ok((!partition.deferrable.is_empty()).then_some(partition.deferrable))
    }

    async fn execute(
        &self,
        session: &Session,
        arguments: Option<&rmcp::model::JsonObject>,
    ) -> CallToolResult {
        let arguments = Value::Object(arguments.cloned().unwrap_or_default());
        let params: ToolSearchParams = match serde_json::from_value(arguments) {
            Ok(params) => params,
            Err(error) => {
                let text = format!("Invalid tool_search arguments: {error}. Pass a query.");
                return CallToolResult::error(vec![ContentBlock::text(text)]);
            }
        };
        let deferrable = match self.deferrable(session).await {
            Ok(Some(deferrable)) => deferrable,
            Ok(None) => {
                let text = "No tools are deferred: every tool is already in your tool list.";
                return CallToolResult::success(vec![ContentBlock::text(text)]);
            }
            Err(error) => {
                let text = format!("Tool search failed: {error}");
                return CallToolResult::error(vec![ContentBlock::text(text)]);
            }
        };
        let messages = session
            .conversation
            .as_ref()
            .map(|conversation| conversation.messages().as_slice())
            .unwrap_or_default();
        let loaded_before = tool_search::loaded_names(messages);
        let outcome = tool_search::search(&params.query, params.max_results, &deferrable);
        tool_search::result(&params.query, &outcome, &loaded_before)
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for ToolSearchOperation {
    fn name(&self) -> &'static str {
        "tool_search"
    }

    async fn inference_tools(&self, session: &Session) -> Result<Vec<Tool>> {
        Ok(match self.deferrable(session).await? {
            Some(_) => vec![tool_search::tool()],
            None => Vec::new(),
        })
    }

    async fn prompt_parts(
        &self,
        session: &Session,
        _conversation: &Conversation,
    ) -> Result<Vec<(String, String)>> {
        Ok(self
            .deferrable(session)
            .await?
            .map(|deferrable| {
                (
                    "deferred_tools".to_string(),
                    tool_search::prompt_section(&deferrable),
                )
            })
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
                    .is_ok_and(|tool_call| tool_call.name.as_ref() == TOOL_SEARCH_TOOL_NAME)
            })
            .collect();
        if pending.is_empty() {
            return not_applicable();
        }

        let mut response = Message::user();
        for (request, disposition) in pending {
            let result: std::result::Result<CallToolResult, ErrorData> = match disposition {
                ToolDisposition::Execute => {
                    let tool_call = request.tool_call.as_ref().map_err(|error| {
                        anyhow!("tool_search tool call could not be parsed: {error}")
                    })?;
                    let span = tool_span(&tool_call.name, &request.id, &session.id);
                    Ok(self
                        .execute(session, tool_call.arguments.as_ref())
                        .instrument(span)
                        .await)
                }
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
