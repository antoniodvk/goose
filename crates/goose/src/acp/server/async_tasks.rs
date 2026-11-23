//! Background tasks of the agent behind a session's provider (ACP agents that
//! report AIR `asyncTasks`), passed on to a client that advertised the same
//! capability, and stopped on its request.

use super::*;
use crate::acp::provider::ASYNC_TASK_STOP_METHOD;
use agent_client_protocol::UntypedMessage;

/// A session's background-task update, by Goose session id.
pub(super) type AsyncTaskUpdate = (String, serde_json::Value);

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StopAsyncTaskParams {
    session_id: String,
    async_task_id: String,
}

/// Whether the client advertised `_meta.jetbrains.air.capabilities` with
/// `asyncTasks`, the same capability an AIR agent waits for before it reports
/// background tasks.
pub(super) fn client_supports_async_tasks(args: &InitializeRequest) -> bool {
    args.client_capabilities
        .meta
        .as_ref()
        .and_then(|meta| meta.get("jetbrains"))
        .and_then(|jetbrains| jetbrains.get("air"))
        .and_then(|air| air.get("capabilities"))
        .and_then(|capabilities| capabilities.as_array())
        .is_some_and(|capabilities| capabilities.iter().any(|c| c == "asyncTasks"))
}

/// The agent's update under the Goose session id, every field unchanged.
fn async_task_notification(
    session_id: &str,
    update: serde_json::Value,
) -> Result<UntypedMessage, agent_client_protocol::Error> {
    UntypedMessage::new(
        "session/update",
        serde_json::json!({ "sessionId": session_id, "update": update }),
    )
}

fn stop_async_task_error(error: ProviderError) -> agent_client_protocol::Error {
    match error {
        ProviderError::NotImplemented(message) => {
            agent_client_protocol::Error::invalid_params().data(message)
        }
        other => agent_client_protocol::Error::internal_error().data(other.to_string()),
    }
}

impl AcpConnection {
    fn supports_async_tasks(&self) -> bool {
        self.client_supports_async_tasks
            .get()
            .copied()
            .unwrap_or(false)
    }

    /// Feeds the background tasks of the session's current provider to the
    /// connection's forwarder. A session registered again with the same
    /// provider (load, reload) keeps its one subscription, so no update is
    /// passed on twice; a new provider (model or provider switch) replaces it,
    /// and the old one ends with its provider.
    pub(super) async fn subscribe_async_tasks(&self, session_id: &str) {
        let Ok(provider) = self.services.provider(session_id).await else {
            return;
        };
        {
            let Ok(mut subscriptions) = self.async_task_subscriptions.lock() else {
                return;
            };
            subscriptions.retain(|_, subscribed| subscribed.strong_count() > 0);
            if subscriptions.get(session_id).is_some_and(|subscribed| {
                std::sync::Weak::ptr_eq(subscribed, &Arc::downgrade(&provider))
            }) {
                return;
            }
            subscriptions.insert(session_id.to_string(), Arc::downgrade(&provider));
        }
        let Some(mut updates) = provider.subscribe_async_tasks() else {
            return;
        };
        drop(provider);
        let session_id = session_id.to_string();
        let tx = self.async_task_update_tx.clone();
        tokio::spawn(async move {
            loop {
                match updates.recv().await {
                    Ok(update) => {
                        if tx.send((session_id.clone(), update)).is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                        warn!(%session_id, skipped, "Background task updates were dropped");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }

    pub(super) async fn start_async_task_forwarder(self: &Arc<Self>, cx: &ConnectionTo<Client>) {
        let Some(mut updates) = self.async_task_update_rx.lock().await.take() else {
            return;
        };
        let server = Arc::downgrade(self);
        let cx = cx.clone();
        tokio::spawn(async move {
            while let Some((session_id, update)) = updates.recv().await {
                let Some(server) = server.upgrade() else {
                    break;
                };
                if !server.supports_async_tasks()
                    || server.closed_session_ids.lock().await.contains(&session_id)
                {
                    continue;
                }
                let sent = async_task_notification(&session_id, update)
                    .and_then(|notification| cx.send_notification(notification));
                if let Err(error) = sent {
                    warn!(%session_id, %error, "Failed to forward a background task update");
                }
            }
        });
    }

    /// `_session/async_task/stop { sessionId, asyncTaskId }` for the agent
    /// behind the session's provider; its response comes back unchanged.
    pub(super) async fn on_stop_async_task(
        &self,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, agent_client_protocol::Error> {
        let params: StopAsyncTaskParams = serde_json::from_value(params).invalid_params_err()?;
        let provider = self
            .services
            .provider(&params.session_id)
            .await
            .invalid_params_err_ctx("Session has no provider")?;
        provider
            .stop_async_task(&params.async_task_id)
            .await
            .map_err(stop_async_task_error)
    }
}

pub(super) fn is_stop_async_task_method(method: &str) -> bool {
    method == ASYNC_TASK_STOP_METHOD
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::ClientCapabilities;

    #[derive(Debug)]
    struct BackgroundTaskProvider {
        updates: tokio::sync::broadcast::Sender<serde_json::Value>,
        stopped: std::sync::Mutex<Vec<String>>,
    }

    impl BackgroundTaskProvider {
        fn new() -> Self {
            Self {
                updates: tokio::sync::broadcast::channel(8).0,
                stopped: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl Provider for BackgroundTaskProvider {
        fn get_name(&self) -> &str {
            "codex-acp"
        }

        fn subscribe_async_tasks(
            &self,
        ) -> Option<tokio::sync::broadcast::Receiver<serde_json::Value>> {
            Some(self.updates.subscribe())
        }

        async fn stop_async_task(
            &self,
            async_task_id: &str,
        ) -> Result<serde_json::Value, ProviderError> {
            self.stopped.lock().unwrap().push(async_task_id.to_string());
            Ok(serde_json::json!({ "stopped": true }))
        }

        async fn stream(
            &self,
            _model_config: &goose_providers::model::ModelConfig,
            _system: &str,
            _messages: &[Message],
            _tools: &[rmcp::model::Tool],
        ) -> Result<crate::providers::base::MessageStream, ProviderError> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[derive(Debug)]
    struct PlainProvider;

    #[async_trait::async_trait]
    impl Provider for PlainProvider {
        fn get_name(&self) -> &str {
            "openai"
        }

        async fn stream(
            &self,
            _model_config: &goose_providers::model::ModelConfig,
            _system: &str,
            _messages: &[Message],
            _tools: &[rmcp::model::Tool],
        ) -> Result<crate::providers::base::MessageStream, ProviderError> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    async fn test_server(root: &std::path::Path) -> Arc<AcpConnection> {
        let services = StateMachineServices::with_config(acp_services_config(
            Arc::new(SessionManager::new(root.to_path_buf())),
            root.to_path_buf(),
            true,
            GoosePlatform::GooseCli,
        ));
        let live_voice = Arc::new(LiveVoiceService::from_config(services.runtime.runs.clone()));
        let provider_factory: AcpProviderFactory = Arc::new(|_provider_name| {
            Box::pin(async { Err(anyhow::anyhow!("unused provider factory")) })
        });
        Arc::new(
            AcpConnection::new(AcpConnectionOptions {
                provider_factory,
                builtin_selection: AcpBuiltinSelection::default(),
                config_dir: root.to_path_buf(),
                services,
                additional_source_roots: Vec::new(),
                session_cwd: None,
                live_voice,
            })
            .await
            .unwrap(),
        )
    }

    async fn register_session(
        server: &AcpConnection,
        root: &std::path::Path,
        provider: Arc<dyn Provider>,
    ) -> String {
        let session = server
            .services
            .session_manager
            .create_session(
                root.to_path_buf(),
                "Background task test".to_string(),
                SessionType::Acp,
                GooseMode::Auto,
            )
            .await
            .unwrap();
        server
            .services
            .update_provider(
                provider,
                goose_providers::model::ModelConfig::new("gpt-5"),
                &session.id,
            )
            .await
            .unwrap();
        server.register_acp_session(session.id.clone()).await;
        session.id
    }

    #[test]
    fn client_async_task_capability_is_read_from_air_meta() {
        let mut meta = serde_json::Map::new();
        meta.insert(
            "jetbrains".to_string(),
            serde_json::json!({ "air": { "version": 1, "capabilities": ["asyncTasks"] } }),
        );
        let advertised = InitializeRequest::new(ProtocolVersion::LATEST)
            .client_capabilities(ClientCapabilities::new().meta(meta));
        assert!(client_supports_async_tasks(&advertised));

        let mut other = serde_json::Map::new();
        other.insert(
            "jetbrains".to_string(),
            serde_json::json!({ "air": { "version": 1, "capabilities": ["nativeSubagentSessions"] } }),
        );
        let without = InitializeRequest::new(ProtocolVersion::LATEST)
            .client_capabilities(ClientCapabilities::new().meta(other));
        assert!(!client_supports_async_tasks(&without));
        assert!(!client_supports_async_tasks(&InitializeRequest::new(
            ProtocolVersion::LATEST
        )));
    }

    #[tokio::test]
    async fn agent_task_update_reaches_the_client_under_the_goose_session_once() {
        let root = tempfile::tempdir().unwrap();
        let server = test_server(root.path()).await;
        let _ = server.client_supports_async_tasks.set(true);
        let provider = Arc::new(BackgroundTaskProvider::new());
        let session_id = register_session(&server, root.path(), provider.clone()).await;
        // Registered again with the same provider, as a reload does.
        server.register_acp_session(session_id.clone()).await;

        let (client_read, server_write) = tokio::io::duplex(64 * 1024);
        let (server_read, client_write) = tokio::io::duplex(64 * 1024);
        let (notification_tx, mut notification_rx) = mpsc::unbounded_channel::<UntypedMessage>();
        let client = tokio::spawn(async move {
            Client
                .builder()
                .on_receive_notification(
                    async move |notification: UntypedMessage, _cx| {
                        let _ = notification_tx.send(notification);
                        Ok(())
                    },
                    agent_client_protocol::on_receive_notification!(),
                )
                .connect_to(ByteStreams::new(
                    client_write.compat_write(),
                    client_read.compat(),
                ))
                .await
        });

        let spawned = serde_json::json!({
            "sessionUpdate": "async_task_spawned",
            "asyncTaskId": "task-1",
            "name": "npm run dev",
            "taskType": "shell",
            "showInTranscript": true,
            "canStop": true,
            "toolCallId": "call-1",
        });
        let server_for_connection = server.clone();
        SacpAgent
            .builder()
            .name("async-task-test")
            .connect_with(
                ByteStreams::new(server_write.compat_write(), server_read.compat()),
                async move |cx: ConnectionTo<Client>| {
                    server_for_connection.start_async_task_forwarder(&cx).await;
                    provider.updates.send(spawned.clone()).unwrap();

                    let wait = std::time::Duration::from_secs(1);
                    let notification = tokio::time::timeout(wait, notification_rx.recv())
                        .await
                        .expect("timed out waiting for the task update")
                        .expect("client notification channel closed");
                    assert_eq!(notification.method(), "session/update");
                    assert_eq!(
                        notification.params(),
                        &serde_json::json!({ "sessionId": session_id, "update": spawned })
                    );
                    assert!(
                        tokio::time::timeout(
                            std::time::Duration::from_millis(200),
                            notification_rx.recv()
                        )
                        .await
                        .is_err(),
                        "the update was passed on more than once"
                    );
                    Ok(())
                },
            )
            .await
            .unwrap();
        client.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn stop_goes_to_the_session_provider_and_returns_its_response() {
        let root = tempfile::tempdir().unwrap();
        let server = test_server(root.path()).await;
        let provider = Arc::new(BackgroundTaskProvider::new());
        let session_id = register_session(&server, root.path(), provider.clone()).await;

        let response = server
            .on_stop_async_task(serde_json::json!({
                "sessionId": session_id,
                "asyncTaskId": "task-1",
            }))
            .await
            .unwrap();
        assert_eq!(response, serde_json::json!({ "stopped": true }));
        assert_eq!(
            *provider.stopped.lock().unwrap(),
            vec!["task-1".to_string()]
        );

        let missing = server
            .on_stop_async_task(serde_json::json!({ "sessionId": session_id }))
            .await
            .unwrap_err();
        assert_eq!(
            missing.code,
            agent_client_protocol::ErrorCode::InvalidParams
        );
    }

    #[tokio::test]
    async fn stop_without_background_tasks_is_invalid_params() {
        let root = tempfile::tempdir().unwrap();
        let server = test_server(root.path()).await;
        let session_id = register_session(&server, root.path(), Arc::new(PlainProvider)).await;

        let error = server
            .on_stop_async_task(serde_json::json!({
                "sessionId": session_id,
                "asyncTaskId": "task-1",
            }))
            .await
            .unwrap_err();
        assert_eq!(error.code, agent_client_protocol::ErrorCode::InvalidParams);
    }
}
