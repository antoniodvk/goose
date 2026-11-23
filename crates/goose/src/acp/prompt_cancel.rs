//! Forwards a goose-side stop to the ACP agent (C411).
//!
//! goose stops a turn by dropping the provider's update stream. Without this the
//! agent never learns about it: it keeps working until its turn ends on its own,
//! and every request queued behind the prompt (model or effort switch, the next
//! prompt) waits that long. ACP's answer is the `session/cancel` notification,
//! after which the agent ends the prompt with `StopReason::Cancelled`.

use agent_client_protocol::schema::v1::{CancelNotification, PromptRequest, PromptResponse};
use agent_client_protocol::{Agent, ConnectionTo};
use std::future::Future;

/// Sends `session/prompt` and waits for its response. Once `stopped` resolves
/// first, sends `session/cancel` for the prompt's session and keeps waiting, so
/// the agent's `Cancelled` response still ends the turn in order.
pub(super) async fn send_prompt_forwarding_stop(
    cx: &ConnectionTo<Agent>,
    request: PromptRequest,
    stopped: impl Future<Output = ()>,
) -> Result<PromptResponse, agent_client_protocol::Error> {
    let session_id = request.session_id.clone();
    let response = cx.send_request(request).block_task();
    tokio::pin!(response);
    tokio::pin!(stopped);

    tokio::select! {
        biased;
        result = &mut response => return result,
        () = &mut stopped => {}
    }

    tracing::info!(session_id = %session_id, "forwarding stop to ACP agent");
    if let Err(error) = cx.send_notification(CancelNotification::new(session_id.clone())) {
        tracing::warn!(session_id = %session_id, %error, "failed to send session/cancel");
    }
    response.await
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::{ContentBlock, SessionId, StopReason, TextContent};
    use agent_client_protocol::{ByteStreams, Client, Responder};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio_util::compat::{TokioAsyncReadCompatExt as _, TokioAsyncWriteCompatExt as _};

    /// Runs `send_prompt_forwarding_stop` against a scripted agent that answers a
    /// prompt only when cancelled, or right away with `EndTurn` when `answer_at_once`.
    /// Returns the stop reason and how many `session/cancel` the agent received.
    async fn run(answer_at_once: bool, stop: bool) -> (StopReason, usize) {
        let (client_read, agent_write) = tokio::io::duplex(64 * 1024);
        let (agent_read, client_write) = tokio::io::duplex(64 * 1024);
        let pending: Arc<Mutex<Option<Responder<PromptResponse>>>> = Arc::new(Mutex::new(None));
        let cancels = Arc::new(AtomicUsize::new(0));

        let agent_pending = pending.clone();
        let agent_cancels = cancels.clone();
        tokio::spawn(async move {
            Agent
                .builder()
                .name("cancel-agent")
                .on_receive_request(
                    async move |_req: PromptRequest, responder: Responder<PromptResponse>, _cx| {
                        if answer_at_once {
                            return responder.respond(PromptResponse::new(StopReason::EndTurn));
                        }
                        *agent_pending.lock().unwrap() = Some(responder);
                        Ok(())
                    },
                    agent_client_protocol::on_receive_request!(),
                )
                .on_receive_notification(
                    async move |_n: CancelNotification, _cx| {
                        agent_cancels.fetch_add(1, Ordering::SeqCst);
                        if let Some(responder) = pending.lock().unwrap().take() {
                            responder.respond(PromptResponse::new(StopReason::Cancelled))?;
                        }
                        Ok(())
                    },
                    agent_client_protocol::on_receive_notification!(),
                )
                .connect_to(ByteStreams::new(
                    agent_write.compat_write(),
                    agent_read.compat(),
                ))
                .await
        });

        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            Client
                .builder()
                .connect_with(
                    ByteStreams::new(client_write.compat_write(), client_read.compat()),
                    async move |cx: ConnectionTo<Agent>| {
                        let request = PromptRequest::new(
                            SessionId::new("agent-session"),
                            vec![ContentBlock::Text(TextContent::new("hi"))],
                        );
                        let stopped = async move {
                            if !stop {
                                std::future::pending::<()>().await;
                            }
                        };
                        let result = send_prompt_forwarding_stop(&cx, request, stopped).await;
                        let _ = result_tx.send(result);
                        Ok(())
                    },
                )
                .await
        });

        let response = tokio::time::timeout(Duration::from_secs(5), result_rx)
            .await
            .expect("prompt did not end in time")
            .expect("client loop ended without a result")
            .expect("prompt failed");
        (response.stop_reason, cancels.load(Ordering::SeqCst))
    }

    #[tokio::test]
    async fn stop_reaches_the_agent_and_its_cancelled_answer_ends_the_prompt() {
        assert_eq!(run(false, true).await, (StopReason::Cancelled, 1));
    }

    #[tokio::test]
    async fn prompt_without_stop_sends_no_cancel() {
        assert_eq!(run(true, false).await, (StopReason::EndTurn, 0));
    }
}
