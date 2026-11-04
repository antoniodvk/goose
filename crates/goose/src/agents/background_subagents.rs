//! Subagents that run detached from the parent's turn (`delegate` with `run_in_background`),
//! like Claude Code's background agents. The parent session's agent keeps them, so they
//! outlive the turn that started them. What they send to the parent (decisions it has to
//! make) and how they ended go out through one event stream; the ACP server forwards it to
//! the client, which brings the result into the parent session as a new message.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use goose_agent::subagent::{SubagentOutcome, SubagentRunner, SubagentUpdate};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::agents::subagent_handler::ForegroundSubagentRunner;
use crate::session::Session;

const ENDED_WITHOUT_OUTCOME: &str = "it ended without an outcome";

#[derive(Debug, Clone)]
pub struct BackgroundSubagentEvent {
    pub parent_session_id: String,
    pub subagent_id: String,
    pub update: SubagentUpdate,
}

#[derive(Default)]
struct State {
    /// Tool calls whose background delegation was started, so a later pass over the same
    /// turn does not start it again.
    started_calls: HashSet<String>,
    /// Running subagent → (parent session, cancellation).
    running: HashMap<String, (String, CancellationToken)>,
}

pub struct BackgroundSubagents {
    state: Arc<Mutex<State>>,
    events_tx: mpsc::UnboundedSender<BackgroundSubagentEvent>,
    events_rx: Mutex<Option<mpsc::UnboundedReceiver<BackgroundSubagentEvent>>>,
}

impl Default for BackgroundSubagents {
    fn default() -> Self {
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        Self {
            state: Arc::default(),
            events_tx,
            events_rx: Mutex::new(Some(events_rx)),
        }
    }
}

impl BackgroundSubagents {
    /// The event stream; only the first caller gets it, so every event is handled once.
    pub fn take_events(&self) -> Option<mpsc::UnboundedReceiver<BackgroundSubagentEvent>> {
        self.events_rx.lock().ok()?.take()
    }

    /// Starts the subagent that `call_id` delegated in the background, once per tool call.
    /// A subagent that is still running is not started a second time.
    pub(crate) async fn start(
        &self,
        runner: &ForegroundSubagentRunner,
        parent: &Session,
        call_id: &str,
        subagent_id: &str,
    ) {
        let cancel = CancellationToken::new();
        {
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            if !state.started_calls.insert(call_id.to_string()) {
                return;
            }
            if state.running.contains_key(subagent_id) {
                tracing::warn!(subagent_id, "background subagent is already running");
                return;
            }
            state
                .running
                .insert(subagent_id.to_string(), (parent.id.clone(), cancel.clone()));
        }

        let mut updates = runner.start(parent, subagent_id, cancel).await;
        let state = Arc::clone(&self.state);
        let events_tx = self.events_tx.clone();
        let parent_session_id = parent.id.clone();
        let subagent_id = subagent_id.to_string();
        tokio::spawn(
            async move {
                let event = |update| BackgroundSubagentEvent {
                    parent_session_id: parent_session_id.clone(),
                    subagent_id: subagent_id.clone(),
                    update,
                };
                let mut outcome = None;
                while let Some(update) = updates.next().await {
                    if let SubagentUpdate::Finished(finished) = update {
                        outcome = Some(finished);
                        break;
                    }
                    let _ = events_tx.send(event(update));
                }
                if let Ok(mut state) = state.lock() {
                    state.running.remove(&subagent_id);
                }
                let outcome = outcome
                    .unwrap_or_else(|| SubagentOutcome::Failed(ENDED_WITHOUT_OUTCOME.to_string()));
                let _ = events_tx.send(event(SubagentUpdate::Finished(outcome)));
            }
            .in_current_span(),
        );
    }

    /// Stops the background subagents of a parent session when it is closed. A running
    /// subagent holds the parent's agent, so nothing else would end it.
    pub fn cancel_for_parent(&self, parent_session_id: &str) {
        if let Ok(state) = self.state.lock() {
            for (parent, cancel) in state.running.values() {
                if parent == parent_session_id {
                    cancel.cancel();
                }
            }
        }
    }
}
