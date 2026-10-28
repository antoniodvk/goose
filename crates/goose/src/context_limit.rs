use anyhow::Result;
use goose_providers::context_limit::ContextLimitResolver;

use crate::config::Config;
use crate::providers::base::Provider;
use crate::session::{ContextWindowState, ExtensionState, Session, SessionManager};

pub async fn get_context_limit(provider: &dyn Provider, model: &str) -> Result<usize> {
    let override_limit = Config::global().get_goose_context_limit()?;
    Ok(provider.get_context_limit(model, override_limit).await)
}

pub fn get_local_context_limit(provider_name: &str, model: &str) -> Result<usize> {
    let override_limit = Config::global().get_goose_context_limit()?;
    let mut configured_limits = Vec::new();

    #[cfg(feature = "aws-providers")]
    if provider_name == "aws_bedrock" {
        if let Some(limit) = crate::providers::bedrock::local_context_limit(model) {
            configured_limits.push((model.to_string(), limit));
        }
    }

    #[cfg(feature = "local-inference")]
    if provider_name == "local" {
        if let Some(limit) = crate::providers::local_inference::local_context_limit(model) {
            configured_limits.push((model.to_string(), limit));
        }
    }

    configured_limits.extend(
        crate::config::declarative_providers::load_provider(provider_name)
            .ok()
            .into_iter()
            .flat_map(|loaded| loaded.config.models)
            .filter_map(|model| model.context_limit.map(|limit| (model.name, limit))),
    );

    Ok(ContextLimitResolver::new(provider_name)
        .with_configured_limits(configured_limits)
        .resolve_local(model, override_limit))
}

/// The context window size to tell clients about, and the state worth remembering.
#[derive(Debug, PartialEq, Eq)]
pub struct ReportedContextLimit {
    /// `0` when the size is unknown everywhere.
    pub size: usize,
    /// Set when the size is known and differs from what the session stored.
    pub remember: Option<ContextWindowState>,
}

/// Decide what to report: a global override wins, then the size the provider
/// knows now, then the size stored for this session and the same model, else `0`.
pub fn reported_context_limit(
    model: &str,
    override_limit: Option<usize>,
    known: Option<usize>,
    stored: Option<&ContextWindowState>,
) -> ReportedContextLimit {
    let stored_size = stored
        .filter(|state| state.model == model)
        .map(|state| state.size);
    let remember = known
        .filter(|size| stored_size != Some(*size))
        .map(|size| ContextWindowState {
            model: model.to_string(),
            size,
        });
    ReportedContextLimit {
        size: override_limit.or(known).or(stored_size).unwrap_or(0),
        remember,
    }
}

/// Context window size for usage notifications to clients: `0` instead of a guess.
/// Remembers a newly learned size in the session.
pub async fn get_reported_context_limit(
    session_manager: &SessionManager,
    session: &Session,
    provider: &dyn Provider,
    model: &str,
) -> Result<usize> {
    let override_limit = Config::global().get_goose_context_limit()?;
    let known = provider.get_known_context_limit(model, None).await;
    let stored = ContextWindowState::from_extension_data(&session.extension_data);
    let reported = reported_context_limit(model, override_limit, known, stored.as_ref());
    if let Some(state) = &reported.remember {
        if let Err(error) = session_manager
            .set_extension_state(&session.id, state)
            .await
        {
            tracing::warn!(session_id = session.id, %error, "Failed to remember the context window size");
        }
    }
    Ok(reported.size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(model: &str, size: usize) -> ContextWindowState {
        ContextWindowState {
            model: model.to_string(),
            size,
        }
    }

    #[test]
    fn unknown_everywhere_reports_zero() {
        let reported = reported_context_limit("m", None, None, None);
        assert_eq!(reported.size, 0);
        assert_eq!(reported.remember, None);
    }

    #[test]
    fn known_size_is_reported_and_remembered_once() {
        let reported = reported_context_limit("m", None, Some(1_000_000), None);
        assert_eq!(reported.size, 1_000_000);
        assert_eq!(reported.remember, Some(stored("m", 1_000_000)));

        let again =
            reported_context_limit("m", None, Some(1_000_000), Some(&stored("m", 1_000_000)));
        assert_eq!(again.size, 1_000_000);
        assert_eq!(again.remember, None);
    }

    #[test]
    fn stored_size_of_the_same_model_fills_in_for_unknown() {
        let reported = reported_context_limit("m", None, None, Some(&stored("m", 1_000_000)));
        assert_eq!(reported.size, 1_000_000);
        assert_eq!(reported.remember, None);
    }

    #[test]
    fn stored_size_of_another_model_is_not_reported() {
        let reported = reported_context_limit("new", None, None, Some(&stored("old", 1_000_000)));
        assert_eq!(reported.size, 0);
        assert_eq!(reported.remember, None);
    }

    #[test]
    fn changed_size_replaces_the_stored_one() {
        let reported =
            reported_context_limit("m", None, Some(200_000), Some(&stored("m", 1_000_000)));
        assert_eq!(reported.size, 200_000);
        assert_eq!(reported.remember, Some(stored("m", 200_000)));
    }

    #[test]
    fn override_wins_but_is_not_remembered() {
        let reported =
            reported_context_limit("m", Some(64_000), None, Some(&stored("m", 1_000_000)));
        assert_eq!(reported.size, 64_000);
        assert_eq!(reported.remember, None);
    }
}
