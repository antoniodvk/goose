use anyhow::Result;
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::acp::{
    extension_configs_to_mcp_servers, AcpProvider, AcpProviderConfig, ACP_CURRENT_MODEL,
};
use crate::config::search_path::SearchPaths;
use crate::config::{Config, GooseMode};
use crate::providers::base::{
    current_working_dir, ConfigKey, ProviderDef, ProviderDescriptor, ProviderMetadata,
};
use crate::providers::catalog::ProviderSetupMetadata;

pub(crate) const ANTIGRAVITY_ACP_PROVIDER_NAME: &str = "antigravity-acp";
const ANTIGRAVITY_ACP_DOC_URL: &str = "https://antigravity.google/docs/cli/acp";
pub(crate) const ANTIGRAVITY_ACP_BINARY: &str = "agy";
const ANTIGRAVITY_COMMAND_KEY: &str = "ANTIGRAVITY_COMMAND";
const GCP_PROJECT_ID_KEY: &str = "GCP_PROJECT_ID";
const GCP_LOCATION_KEY: &str = "GCP_LOCATION";
const DEFAULT_GCP_LOCATION: &str = "global";

pub struct AntigravityAcpProvider;

impl goose_providers::base::ProviderDescriptor for AntigravityAcpProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            ANTIGRAVITY_ACP_PROVIDER_NAME,
            "Google Antigravity (Vertex AI)",
            "Use goose with Antigravity's Gemini agent via the agy ACP agent, billed to a Google Cloud project.",
            ACP_CURRENT_MODEL,
            vec![],
            ANTIGRAVITY_ACP_DOC_URL,
            vec![
                ConfigKey::new(GCP_PROJECT_ID_KEY, true, false, None, true),
                ConfigKey::new(GCP_LOCATION_KEY, false, false, Some(DEFAULT_GCP_LOCATION), false),
                ConfigKey::new(ANTIGRAVITY_COMMAND_KEY, false, false, None, false),
            ],
        )
        .with_setup_steps(vec![
            "Install the Antigravity CLI, which provides the `agy` agent",
            "Run `gcloud auth application-default login` so application default credentials exist",
            "Set GCP_PROJECT_ID to the Google Cloud project that pays for the requests",
        ])
        .with_setup(
            ProviderSetupMetadata::cli_agent(ANTIGRAVITY_ACP_BINARY, &["antigravity-acp", "agy"])
                .with_acp()
                .with_docs_url(ANTIGRAVITY_ACP_DOC_URL)
                .with_capabilities(true, true, true),
        )
    }
}

impl ProviderDef for AntigravityAcpProvider {
    type Provider = AcpProvider;
    const SESSION_BOUND: bool = true;
    const RUNS_OWN_TOOL_LOOP: bool = true;

    fn from_env(
        extensions: Vec<crate::config::ExtensionConfig>,
        tls_config: Option<crate::providers::api_client::TlsConfig>,
    ) -> BoxFuture<'static, Result<AcpProvider>> {
        Self::from_env_with_working_dir(extensions, current_working_dir(), tls_config)
    }

    fn from_env_with_working_dir(
        extensions: Vec<crate::config::ExtensionConfig>,
        working_dir: PathBuf,
        _tls_config: Option<crate::providers::api_client::TlsConfig>,
    ) -> BoxFuture<'static, Result<AcpProvider>> {
        Box::pin(async move {
            let config = Config::global();
            let resolved_command = resolve_command(config)?;
            let project = config.get_param::<String>(GCP_PROJECT_ID_KEY)?;
            let location = config
                .get_param::<String>(GCP_LOCATION_KEY)
                .unwrap_or_else(|_| DEFAULT_GCP_LOCATION.to_string());
            let goose_mode = config.resolve_goose_mode()?;

            let mode_mapping = HashMap::from([
                // "auto" runs tools without asking, closest to autonomous mode.
                (GooseMode::Auto, vec!["auto".to_string()]),
                // "default" asks before file edits and commands.
                (GooseMode::Approve, vec!["default".to_string()]),
                // "accept-edits" accepts file edits but still asks for risky commands.
                (GooseMode::SmartApprove, vec!["accept-edits".to_string()]),
                // "plan" keeps the agent read-only, aligning with chat-only intent.
                (GooseMode::Chat, vec!["plan".to_string()]),
            ]);

            let provider_config = AcpProviderConfig {
                command: resolved_command,
                args: vec![
                    // goose already runs one agent process per session.
                    "acp".to_string(),
                    "--disable-telemetry".to_string(),
                ],
                env: vec![
                    // Authenticate through application default credentials, so the
                    // request is billed to the Google Cloud project below.
                    ("AGY_ADC_AUTH".to_string(), "true".to_string()),
                    ("GOOGLE_CLOUD_PROJECT".to_string(), project.clone()),
                    ("GOOGLE_CLOUD_QUOTA_PROJECT".to_string(), project.clone()),
                    ("GOOGLE_CLOUD_LOCATION".to_string(), location),
                ],
                // Keep the agent on Vertex AI; inherited API keys would switch it
                // to the Gemini API and bill a different account.
                env_remove: vec!["GEMINI_API_KEY".to_string(), "GOOGLE_API_KEY".to_string()],
                work_dir: working_dir,
                mcp_servers: extension_configs_to_mcp_servers(&extensions),
                session_mode_id: mode_mapping[&goose_mode].first().cloned(),
                session_config_options: vec![],
                // agy advertises the model as a "model" select config option.
                model_config_option_id: Some("model".to_string()),
                mode_mapping,
                notification_callback: None,
            };

            let metadata = Self::metadata();
            AcpProvider::connect(metadata.name, goose_mode, provider_config).await
        })
    }
}

fn resolve_command(config: &Config) -> Result<PathBuf> {
    if let Ok(command) = config.get_param::<String>(ANTIGRAVITY_COMMAND_KEY) {
        let command = command.trim();
        if !command.is_empty() {
            return Ok(PathBuf::from(command));
        }
    }
    if let Some(installed) = installed_executable() {
        return Ok(installed);
    }
    SearchPaths::builder().resolve(ANTIGRAVITY_ACP_BINARY)
}

/// The Antigravity installer puts `agy` outside the PATH goose sees.
fn installed_executable() -> Option<PathBuf> {
    let local_app_data = std::env::var_os("LOCALAPPDATA")?;
    let executable = Path::new(&local_app_data)
        .join("agy")
        .join("bin")
        .join("agy.exe");
    executable.is_file().then_some(executable)
}
