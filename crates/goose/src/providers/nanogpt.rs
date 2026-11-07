use super::api_client::{ApiClient, AuthMethod};
use super::base::{ConfigKey, MessageStream, Provider, ProviderDef, ProviderMetadata};
use super::openai_compatible::{handle_status, stream_openai_compat};
use super::retry::ProviderRetry;
use crate::conversation::message::Message;
use anyhow::Result;
use async_trait::async_trait;
use futures::future::BoxFuture;
use goose_providers::errors::ProviderError;
use goose_providers::formats::openai::{
    create_request, nearest_reasoning_effort, reasoning_effort_for_thinking_in_catalog,
};
use goose_providers::http_status::read_json_response;
use goose_providers::images::ImageFormat;
use goose_providers::model::ModelConfig;
use goose_providers::request_log::{start_log, LoggerHandleExt};
use goose_providers::usage::{parse_nanogpt_usage, AccountUsage};
use rmcp::model::Tool;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub const NANOGPT_PROVIDER_NAME: &str = "nano-gpt";
pub const NANOGPT_API_HOST: &str = "https://nano-gpt.com/api/v1";
pub const NANOGPT_SUBSCRIPTION_HOST: &str = "https://nano-gpt.com/api/subscription/v1";
pub const NANOGPT_DEFAULT_MODEL: &str = "anthropic/claude-sonnet-4.6";
const NANOGPT_ACCOUNT_HOST: &str = "https://nano-gpt.com/api";
pub const NANOGPT_DOC_URL: &str = "https://docs.nano-gpt.com/";
const NANOGPT_API_KEY: &str = "NANOGPT_API_KEY";

#[derive(serde::Serialize)]
pub struct NanoGptProvider {
    #[serde(skip)]
    api_client: ApiClient,
    #[serde(skip)]
    name: String,
    /// `reasoning_efforts` per model from NanoGPT's own model list (N485).
    #[serde(skip)]
    reasoning_efforts: Arc<RwLock<HashMap<String, Vec<String>>>>,
    /// For the account endpoints, which sit on other hosts than `api_client`.
    #[serde(skip)]
    api_key: String,
    #[serde(skip)]
    tls_config: Option<crate::providers::api_client::TlsConfig>,
}

impl NanoGptProvider {
    /// Subscription quotas (`GET subscription/v1/usage`, inference key) and the
    /// cash balance (`POST check-balance`, `x-api-key`), see
    /// `goose_providers::usage::parse_nanogpt_usage` for the sources.
    async fn fetch_account_usage(
        api_key: &str,
        tls_config: Option<crate::providers::api_client::TlsConfig>,
        subscription_host: &str,
        account_host: &str,
    ) -> Result<AccountUsage, ProviderError> {
        let subscription = Self::build_client(subscription_host, api_key, tls_config.clone())?
            .response_get("usage")
            .await?;
        let subscription: serde_json::Value =
            read_json_response(handle_status(subscription).await?).await?;

        // The documentation names no request body for check-balance.
        let balance = Self::build_client(account_host, api_key, tls_config)?
            .with_header("x-api-key", api_key)?
            .response_post("check-balance", &serde_json::json!({}))
            .await?;
        let balance: serde_json::Value = read_json_response(handle_status(balance).await?).await?;

        parse_nanogpt_usage(&subscription, &balance)
    }

    fn build_client(
        host: &str,
        api_key: &str,
        tls_config: Option<crate::providers::api_client::TlsConfig>,
    ) -> Result<ApiClient> {
        ApiClient::new_with_tls(
            host.to_string(),
            AuthMethod::BearerToken(api_key.to_string()),
            tls_config,
        )?
        .with_header("x-client", "goose")
    }

    async fn check_subscription(
        api_key: &str,
        tls_config: Option<crate::providers::api_client::TlsConfig>,
    ) -> bool {
        let client = match Self::build_client(NANOGPT_SUBSCRIPTION_HOST, api_key, tls_config) {
            Ok(c) => c,
            Err(_) => return false,
        };

        match client.response_get("usage").await {
            Ok(resp) => resp
                .json::<serde_json::Value>()
                .await
                .ok()
                .and_then(|json| json.get("active")?.as_bool())
                .unwrap_or(false),
            Err(_) => false,
        }
    }

    /// Models from the `models?detailed=true` answer that take tools, and the
    /// effort levels each of them declares.
    fn parse_models(json: &serde_json::Value) -> Result<Vec<serde_json::Value>, ProviderError> {
        if let Some(err_obj) = json.get("error") {
            let msg = err_obj
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            return Err(ProviderError::RequestFailed(format!(
                "NanoGPT API returned an error: {}",
                msg
            )));
        }

        json.get("data")
            .and_then(|v| v.as_array())
            .cloned()
            .ok_or_else(|| {
                ProviderError::RequestFailed("Missing 'data' field in JSON response".into())
            })
    }

    fn store_reasoning_efforts(&self, data: &[serde_json::Value]) {
        let efforts: HashMap<String, Vec<String>> = data
            .iter()
            .filter_map(|model| {
                let id = model.get("id")?.as_str()?;
                let levels: Vec<String> = model
                    .get("reasoning_efforts")?
                    .as_array()?
                    .iter()
                    .filter_map(|level| level.as_str().map(str::to_string))
                    .collect();
                (!levels.is_empty()).then(|| (id.to_string(), levels))
            })
            .collect();
        if let Ok(mut guard) = self.reasoning_efforts.write() {
            *guard = efforts;
        }
    }

    async fn fetch_model_list(&self) -> Result<Vec<serde_json::Value>, ProviderError> {
        let response = self
            .api_client
            .request("models?detailed=true")
            .response_get()
            .await
            .map_err(|e| {
                ProviderError::RequestFailed(format!(
                    "Failed to fetch models from NanoGPT API: {}",
                    e
                ))
            })?;

        let json: serde_json::Value = response.json().await.map_err(|e| {
            ProviderError::RequestFailed(format!(
                "Failed to parse NanoGPT models API response as JSON: {}",
                e
            ))
        })?;

        let data = Self::parse_models(&json)?;
        self.store_reasoning_efforts(&data);
        Ok(data)
    }

    pub async fn from_env(
        tls_config: Option<crate::providers::api_client::TlsConfig>,
    ) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret(NANOGPT_API_KEY)?;

        let is_subscription = Self::check_subscription(&api_key, tls_config.clone()).await;
        let host = if is_subscription {
            tracing::debug!("NanoGPT subscription active, using subscription endpoint");
            NANOGPT_SUBSCRIPTION_HOST.to_string()
        } else {
            tracing::debug!("NanoGPT using pay-as-you-go endpoint");
            NANOGPT_API_HOST.to_string()
        };

        let api_client = Self::build_client(&host, &api_key, tls_config.clone())?
            .with_request_builder(crate::session_context::session_id_request_builder());

        let provider = Self {
            api_client,
            name: NANOGPT_PROVIDER_NAME.to_string(),
            reasoning_efforts: Arc::default(),
            api_key,
            tls_config,
        };
        // The chat's effort menu is built from this list; without it goose
        // falls back to its catalog.
        if let Err(error) = provider.fetch_model_list().await {
            tracing::debug!("NanoGPT effort levels unavailable: {}", error);
        }
        Ok(provider)
    }
}

impl goose_providers::base::ProviderDescriptor for NanoGptProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            NANOGPT_PROVIDER_NAME,
            "NanoGPT",
            "Access multiple AI models through NanoGPT's unified API",
            NANOGPT_DEFAULT_MODEL,
            vec![NANOGPT_DEFAULT_MODEL],
            NANOGPT_DOC_URL,
            vec![ConfigKey::new(NANOGPT_API_KEY, true, true, None, true)],
        )
    }
}

impl ProviderDef for NanoGptProvider {
    type Provider = Self;

    fn from_env(
        _extensions: Vec<crate::config::ExtensionConfig>,
        tls_config: Option<crate::providers::api_client::TlsConfig>,
    ) -> BoxFuture<'static, Result<Self::Provider>> {
        Box::pin(Self::from_env(tls_config))
    }
}

#[async_trait]
impl Provider for NanoGptProvider {
    fn get_name(&self) -> &str {
        &self.name
    }

    async fn fetch_supported_models(&self) -> Result<Vec<String>, ProviderError> {
        let data = self.fetch_model_list().await?;

        let mut models: Vec<String> = data
            .iter()
            .filter_map(|model| {
                let id = model.get("id").and_then(|v| v.as_str())?;
                let supports_tool_calling = model
                    .get("capabilities")
                    .and_then(|c| c.get("tool_calling"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if supports_tool_calling {
                    Some(id.to_string())
                } else {
                    None
                }
            })
            .collect();

        models.sort();
        Ok(models)
    }

    async fn fetch_usage(&self) -> Result<Option<AccountUsage>, ProviderError> {
        Self::fetch_account_usage(
            &self.api_key,
            self.tls_config.clone(),
            NANOGPT_SUBSCRIPTION_HOST,
            NANOGPT_ACCOUNT_HOST,
        )
        .await
        .map(Some)
    }

    /// NanoGPT's own model list, else the NanoGPT catalog entry: the same
    /// levels `stream` sends from (N485/N774).
    fn declared_reasoning_efforts(&self, model_name: &str) -> Option<Vec<String>> {
        self.reasoning_efforts
            .read()
            .ok()
            .and_then(|levels| levels.get(model_name).cloned())
            .or_else(|| {
                goose_providers::formats::openai::catalog_reasoning_efforts(
                    NANOGPT_PROVIDER_NAME,
                    model_name,
                )
                .filter(|levels| !levels.is_empty())
            })
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let mut payload = create_request(
            model_config,
            system,
            messages,
            tools,
            &ImageFormat::OpenAi,
            true,
        )?;

        // create_request clamps reasoning_effort against the OpenAI catalog; NanoGPT
        // takes it for every reasoning model (N460), with the levels its own
        // model list declares (N485).
        if model_config.is_reasoning_model() {
            let declared = self.declared_reasoning_efforts(&model_config.model_name);
            if let Some(effort) =
                model_config
                    .thinking_effort()
                    .and_then(|effort| match &declared {
                        Some(levels) => {
                            nearest_reasoning_effort(Some(levels), &model_config.model_name, effort)
                        }
                        None => reasoning_effort_for_thinking_in_catalog(
                            NANOGPT_PROVIDER_NAME,
                            &model_config.model_name,
                            effort,
                        ),
                    })
            {
                payload["reasoning_effort"] = serde_json::json!(effort);
            }
        }

        let mut log = start_log(model_config, &payload)?;

        let response = self
            .with_retry(|| async {
                let resp = self
                    .api_client
                    .request("chat/completions")
                    .model_headers(model_config)?
                    .streaming(true)
                    .response_post(&payload)
                    .await?;
                handle_status(resp).await
            })
            .await
            .inspect_err(|e| {
                let _ = log.error(e);
            })?;

        stream_openai_compat(response, log)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goose_providers::base::ProviderDescriptor as _;

    #[tokio::test]
    async fn account_usage_combines_subscription_quotas_and_balance() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/subscription/v1/usage"))
            .and(header("authorization", "Bearer nano-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "active": true,
                "state": "active",
                "limits": { "dailyInputTokens": 1000000, "weeklyInputTokens": null, "dailyImages": null },
                "dailyInputTokens": { "used": 250000, "remaining": 750000, "percentUsed": 0.25, "resetAt": 1788652800000_i64 },
                "weeklyInputTokens": null,
                "dailyImages": null,
                "period": { "currentPeriodEnd": null }
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/check-balance"))
            .and(header("x-api-key", "nano-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "usd_balance": "129.46956147", "nano_balance": "26.71801147"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let usage = NanoGptProvider::fetch_account_usage(
            "nano-key",
            None,
            &format!("{}/api/subscription/v1", server.uri()),
            &format!("{}/api", server.uri()),
        )
        .await
        .unwrap();

        assert_eq!(usage.windows[0].used_percent, Some(25.0));
        assert_eq!(usage.balances[0].amount, 129.46956147);
    }

    #[tokio::test]
    async fn account_usage_maps_a_rejected_key_to_authentication() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/subscription/v1/usage"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "error": { "message": "invalid_api_key" }
            })))
            .mount(&server)
            .await;

        let error = NanoGptProvider::fetch_account_usage(
            "wrong",
            None,
            &format!("{}/api/subscription/v1", server.uri()),
            &format!("{}/api", server.uri()),
        )
        .await
        .unwrap_err();

        assert!(matches!(error, ProviderError::Authentication(_)));
    }

    #[test]
    fn test_metadata() {
        let metadata = NanoGptProvider::metadata();
        assert_eq!(metadata.name, "nano-gpt");
        assert_eq!(metadata.default_model, "anthropic/claude-sonnet-4.6");
        assert_eq!(metadata.config_keys[0].name, NANOGPT_API_KEY);
        assert!(metadata.config_keys[0].required);
        assert!(metadata.config_keys[0].secret);
    }
}
