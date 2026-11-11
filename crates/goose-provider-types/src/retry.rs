use crate::base::Provider;
use crate::errors::ProviderError;
use crate::maybe_send::MaybeSend;
use async_trait::async_trait;
use std::future::Future;
use std::time::Duration;

// wasm32 hosts run without a tokio runtime, so back off on the host's timer.
#[cfg(target_arch = "wasm32")]
use gloo_timers::future::sleep;
#[cfg(not(target_arch = "wasm32"))]
use tokio::time::sleep;

pub const DEFAULT_MAX_RETRIES: usize = 3;
pub const DEFAULT_INITIAL_RETRY_INTERVAL_MS: u64 = 1000;
pub const DEFAULT_BACKOFF_MULTIPLIER: f64 = 2.0;
pub const DEFAULT_MAX_RETRY_INTERVAL_MS: u64 = 30_000;

/// A provider request failed transiently and is about to be retried after `delay`.
#[derive(Debug, Clone)]
pub struct RetryNotice {
    pub attempt: usize,
    pub max_retries: usize,
    pub delay: Duration,
    /// `rate_limit`, `server_error` or `network`.
    pub reason: &'static str,
}

pub type RetryNoticeFn = std::sync::Arc<dyn Fn(&RetryNotice) + Send + Sync>;

#[cfg(not(target_arch = "wasm32"))]
tokio::task_local! {
    /// Set by the ACP server for the duration of a turn so silent provider
    /// retries become visible to the client.
    static RETRY_NOTICE: RetryNoticeFn;
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn with_retry_notice<F>(notify: RetryNoticeFn, fut: F) -> F::Output
where
    F: Future,
{
    RETRY_NOTICE.scope(notify, fut).await
}

/// Reports an upcoming retry to the turn's listener, if any.
pub fn notify_retry(attempt: usize, max_retries: usize, delay: Duration, error: &ProviderError) {
    let reason = match error {
        ProviderError::RateLimitExceeded { .. } => "rate_limit",
        ProviderError::ServerError(_) => "server_error",
        ProviderError::NetworkError(_) => "network",
        _ => return,
    };
    #[cfg(not(target_arch = "wasm32"))]
    let _ = RETRY_NOTICE.try_with(|notify| {
        notify(&RetryNotice {
            attempt,
            max_retries,
            delay,
            reason,
        })
    });
    #[cfg(target_arch = "wasm32")]
    let _ = (attempt, max_retries, delay, reason);
}

/// A provider has left a request unanswered for `waited`. `answered` marks the
/// answer that finally arrived after at least one such notice.
#[derive(Debug, Clone)]
pub struct WaitNotice {
    pub provider: String,
    pub waited: Duration,
    pub answered: bool,
}

pub type WaitNoticeFn = std::sync::Arc<dyn Fn(&WaitNotice) + Send + Sync>;

pub const WAIT_NOTICE_AFTER: Duration = Duration::from_secs(30);
pub const WAIT_NOTICE_EVERY: Duration = Duration::from_secs(10);

#[cfg(not(target_arch = "wasm32"))]
tokio::task_local! {
    /// Set by the ACP server for the duration of a turn so a silent provider
    /// becomes visible to the client.
    static WAIT_NOTICE: WaitNoticeFn;
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn with_wait_notice<F>(notify: WaitNoticeFn, fut: F) -> F::Output
where
    F: Future,
{
    WAIT_NOTICE.scope(notify, fut).await
}

/// Awaits a provider request and reports to the turn's listener while it stays
/// unanswered: first after [`WAIT_NOTICE_AFTER`], then every [`WAIT_NOTICE_EVERY`].
/// The request itself is never cut short.
pub async fn notify_while_waiting<T, E, F>(provider: &str, fut: F) -> Result<T, E>
where
    F: Future<Output = Result<T, E>>,
{
    notify_while_waiting_every(provider, WAIT_NOTICE_AFTER, WAIT_NOTICE_EVERY, fut).await
}

#[cfg(not(target_arch = "wasm32"))]
async fn notify_while_waiting_every<T, E, F>(
    provider: &str,
    after: Duration,
    every: Duration,
    fut: F,
) -> Result<T, E>
where
    F: Future<Output = Result<T, E>>,
{
    let send = |waited: Duration, answered: bool| {
        let _ = WAIT_NOTICE.try_with(|notify| {
            notify(&WaitNotice {
                provider: provider.to_string(),
                waited,
                answered,
            })
        });
    };
    let started = tokio::time::Instant::now();
    let mut fut = std::pin::pin!(fut);
    let mut waited = after;
    let mut notified = false;
    loop {
        match tokio::time::timeout_at(started + waited, fut.as_mut()).await {
            Ok(result) => {
                if notified && result.is_ok() {
                    send(started.elapsed(), true);
                }
                return result;
            }
            Err(_) => {
                notified = true;
                send(waited, false);
                waited += every;
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn notify_while_waiting_every<T, E, F>(
    _provider: &str,
    _after: Duration,
    _every: Duration,
    fut: F,
) -> Result<T, E>
where
    F: Future<Output = Result<T, E>>,
{
    fut.await
}

#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Initial interval between retries in milliseconds
    pub initial_interval_ms: u64,
    /// Multiplier for backoff (exponential)
    pub backoff_multiplier: f64,
    /// Maximum interval between retries in milliseconds
    pub max_interval_ms: u64,
    /// When true, only retry on transient errors (ServerError, NetworkError,
    /// RateLimitExceeded). RequestFailed (4xx client errors) will not be retried.
    pub transient_only: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: DEFAULT_MAX_RETRIES,
            initial_interval_ms: DEFAULT_INITIAL_RETRY_INTERVAL_MS,
            backoff_multiplier: DEFAULT_BACKOFF_MULTIPLIER,
            max_interval_ms: DEFAULT_MAX_RETRY_INTERVAL_MS,
            transient_only: false,
        }
    }
}

impl RetryConfig {
    pub fn new(
        max_retries: usize,
        initial_interval_ms: u64,
        backoff_multiplier: f64,
        max_interval_ms: u64,
    ) -> Self {
        Self {
            max_retries,
            initial_interval_ms,
            backoff_multiplier,
            max_interval_ms,
            transient_only: false,
        }
    }

    pub fn transient_only(mut self) -> Self {
        self.transient_only = true;
        self
    }

    pub fn max_retries(&self) -> usize {
        self.max_retries
    }

    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        if attempt == 0 {
            return Duration::from_millis(0);
        }

        let exponent = (attempt - 1) as u32;
        let base_delay_ms = (self.initial_interval_ms as f64
            * self.backoff_multiplier.powi(exponent as i32)) as u64;

        let capped_delay_ms = std::cmp::min(base_delay_ms, self.max_interval_ms);

        let jitter_factor_to_avoid_thundering_herd = 0.8 + (rand::random::<f64>() * 0.4);
        let jitter_delay_ms =
            (capped_delay_ms as f64 * jitter_factor_to_avoid_thundering_herd) as u64;

        Duration::from_millis(jitter_delay_ms)
    }
}

/// Substrings marking a `RequestFailed` (4xx) as deterministically permanent:
/// Anthropic rejects signed `thinking`/`redacted_thinking` blocks as immutable
/// once a thinking model's config changes mid-conversation, and the identical
/// payload is rebuilt on every retry — so retrying can never succeed.
const PERMANENT_REQUEST_FAILURE_MARKERS: &[&str] = &[
    "blocks in the latest assistant message cannot be modified",
    "must remain as they were in the original response",
    "Reasoning is mandatory for this endpoint",
];

fn is_permanent_request_failure(message: &str) -> bool {
    PERMANENT_REQUEST_FAILURE_MARKERS
        .iter()
        .any(|marker| message.contains(marker))
        || crate::formats::anthropic::is_thinking_signature_error(message)
}

pub fn should_retry(error: &ProviderError, config: &RetryConfig) -> bool {
    match error {
        ProviderError::RateLimitExceeded { .. }
        | ProviderError::ServerError(_)
        | ProviderError::NetworkError(_) => true,
        ProviderError::RequestFailed(message) if is_permanent_request_failure(message) => false,
        ProviderError::RequestFailed(_) => !config.transient_only,
        _ => false,
    }
}

pub async fn retry_operation<F, Fut, T>(
    config: &RetryConfig,
    operation: F,
) -> Result<T, ProviderError>
where
    F: Fn() -> Fut + MaybeSend,
    Fut: Future<Output = Result<T, ProviderError>> + MaybeSend,
    T: MaybeSend,
{
    let mut attempts = 0;

    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                if should_retry(&error, config) && attempts < config.max_retries {
                    attempts += 1;
                    tracing::warn!(
                        "Request failed, retrying ({}/{}): {:?}",
                        attempts,
                        config.max_retries,
                        error
                    );

                    let delay = match &error {
                        ProviderError::RateLimitExceeded {
                            retry_delay: Some(d),
                            ..
                        } => *d,
                        _ => config.delay_for_attempt(attempts),
                    };

                    notify_retry(attempts, config.max_retries, delay, &error);
                    sleep(delay).await;
                    continue;
                }
                return Err(error);
            }
        }
    }
}

/// Trait for retry functionality to keep Provider dyn-compatible.
///
/// All `Provider` implementors get this via the blanket impl below.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait ProviderRetry {
    fn retry_config(&self) -> RetryConfig {
        RetryConfig::default()
    }

    async fn with_retry<F, Fut, T>(&self, operation: F) -> Result<T, ProviderError>
    where
        F: Fn() -> Fut + MaybeSend,
        Fut: Future<Output = Result<T, ProviderError>> + MaybeSend,
        T: MaybeSend,
    {
        self.with_retry_config(operation, self.retry_config()).await
    }

    async fn with_retry_config<F, Fut, T>(
        &self,
        operation: F,
        config: RetryConfig,
    ) -> Result<T, ProviderError>
    where
        F: Fn() -> Fut + MaybeSend,
        Fut: Future<Output = Result<T, ProviderError>> + MaybeSend,
        T: MaybeSend;
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<P: Provider> ProviderRetry for P {
    fn retry_config(&self) -> RetryConfig {
        Provider::retry_config(self)
    }

    async fn with_retry_config<F, Fut, T>(
        &self,
        operation: F,
        config: RetryConfig,
    ) -> Result<T, ProviderError>
    where
        F: Fn() -> Fut + MaybeSend,
        Fut: Future<Output = Result<T, ProviderError>> + MaybeSend,
        T: MaybeSend,
    {
        let mut attempts = 0;
        let mut auth_retried = false;

        loop {
            return match operation().await {
                Ok(result) => Ok(result),
                Err(error) => {
                    // Auth retry is separate from transient-error retries: we get
                    // at most 1 credential refresh, independent of max_retries.
                    if matches!(error, ProviderError::Authentication(_)) && !auth_retried {
                        auth_retried = true;
                        match self.refresh_credentials().await {
                            Ok(()) => {
                                tracing::warn!(
                                    "Credentials refreshed after auth error, retrying: {:?}",
                                    error
                                );
                                continue;
                            }
                            Err(refresh_err) => {
                                tracing::warn!(
                                    "Credential refresh failed, returning original auth error: {:?}",
                                    refresh_err
                                );
                            }
                        }
                    }

                    if should_retry(&error, &config) && attempts < config.max_retries {
                        attempts += 1;
                        tracing::warn!(
                            "Request failed, retrying ({}/{}): {:?}",
                            attempts,
                            config.max_retries,
                            error
                        );

                        let delay = match &error {
                            ProviderError::RateLimitExceeded {
                                retry_delay: Some(provider_delay),
                                ..
                            } => *provider_delay,
                            _ => config.delay_for_attempt(attempts),
                        };

                        let skip_backoff = std::env::var("GOOSE_PROVIDER_SKIP_BACKOFF")
                            .unwrap_or_default()
                            .parse::<bool>()
                            .unwrap_or(false);

                        if skip_backoff {
                            tracing::info!("Skipping backoff due to GOOSE_PROVIDER_SKIP_BACKOFF");
                        } else {
                            notify_retry(attempts, config.max_retries, delay, &error);
                            tracing::info!("Backing off for {:?} before retry", delay);
                            sleep(delay).await;
                        }
                        continue;
                    }

                    Err(error)
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_notices<F>(fut: F) -> (Result<u8, ()>, Vec<(u64, bool)>)
    where
        F: Future<Output = Result<u8, ()>>,
    {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = seen.clone();
        let notify: WaitNoticeFn = std::sync::Arc::new(move |notice: &WaitNotice| {
            assert_eq!(notice.provider, "gcp_vertex_ai");
            sink.lock()
                .unwrap()
                .push((notice.waited.as_millis() as u64, notice.answered));
        });
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let result = runtime.block_on(with_wait_notice(
            notify,
            notify_while_waiting_every(
                "gcp_vertex_ai",
                Duration::from_millis(60),
                Duration::from_millis(40),
                fut,
            ),
        ));
        let seen = seen.lock().unwrap().clone();
        (result, seen)
    }

    #[test]
    fn quick_answer_sends_no_wait_notice() {
        let (result, seen) = wait_notices(async { Ok(1) });
        assert_eq!(result, Ok(1));
        assert!(seen.is_empty());
    }

    #[test]
    fn silent_provider_is_reported_until_it_answers() {
        let (result, seen) = wait_notices(async {
            sleep(Duration::from_millis(130)).await;
            Ok(1)
        });
        assert_eq!(result, Ok(1));
        assert_eq!(seen[..2], [(60, false), (100, false)]);
        let (waited, answered) = *seen.last().unwrap();
        assert!(answered && waited >= 130);
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn late_error_is_not_reported_as_an_answer() {
        let (result, seen) = wait_notices(async {
            sleep(Duration::from_millis(80)).await;
            Err(())
        });
        assert_eq!(result, Err(()));
        assert_eq!(seen, [(60, false)]);
    }

    #[test]
    fn default_config_retries_request_failed() {
        let config = RetryConfig::default();
        let error = ProviderError::RequestFailed("Bad request (400): model not found".into());
        assert!(should_retry(&error, &config));
    }

    #[test]
    fn never_retries_permanent_thinking_block_400() {
        let config = RetryConfig::default();
        let error = ProviderError::RequestFailed(
            "Bad request (400): {\"message\":\"messages.3.content.1: `thinking` or \
             `redacted_thinking` blocks in the latest assistant message cannot be \
             modified. These blocks must remain as they were in the original \
             response.\"}"
                .into(),
        );
        assert!(!should_retry(&error, &config));
    }

    #[test]
    fn permanent_request_failure_marker_detection() {
        assert!(is_permanent_request_failure(
            "messages.3.content.1: `thinking` blocks in the latest assistant message \
             cannot be modified"
        ));
        assert!(is_permanent_request_failure(
            "These blocks must remain as they were in the original response."
        ));
        assert!(is_permanent_request_failure(
            "messages.1.content.0: Invalid `signature` in `thinking` block."
        ));
        assert!(!is_permanent_request_failure(
            "Bad request (400): model not found"
        ));
    }

    #[test]
    fn transient_only_skips_request_failed() {
        let config = RetryConfig::default().transient_only();
        let error = ProviderError::RequestFailed("Bad request (400): model not found".into());
        assert!(!should_retry(&error, &config));
    }

    #[test]
    fn transient_only_still_retries_server_error() {
        let config = RetryConfig::default().transient_only();
        assert!(should_retry(
            &ProviderError::ServerError("500 internal".into()),
            &config
        ));
    }

    #[test]
    fn transient_only_still_retries_network_error() {
        let config = RetryConfig::default().transient_only();
        assert!(should_retry(
            &ProviderError::NetworkError("connection refused".into()),
            &config
        ));
    }

    #[test]
    fn transient_only_still_retries_rate_limit() {
        let config = RetryConfig::default().transient_only();
        assert!(should_retry(
            &ProviderError::RateLimitExceeded {
                details: "too many requests".into(),
                retry_delay: None,
            },
            &config
        ));
    }

    #[test]
    fn never_retries_auth_errors() {
        let config = RetryConfig::default();
        assert!(!should_retry(
            &ProviderError::Authentication("invalid key".into()),
            &config
        ));
    }
}
