//! Helpers on providers that bring their own tools (claude-acp and other ACP agents) never receive
//! goose's built-in `final_output` tool, so the summary every delegated helper owes could never be
//! collected and each such helper failed. A helper on such a provider answers in plain text instead:
//! the final-output step stays out of its turn, and its last answer becomes its result in the same
//! `{"summary": …}` shape the tool would have produced. Main sessions keep the hard error, because a
//! recipe run there explicitly asks for structured output.

use crate::agents::StateMachineServices;
use crate::conversation::message::Message;
use crate::providers::base::Provider;
use crate::session::{Session, SessionType};
use rmcp::model::Role;

/// True for a helper whose provider cannot receive goose's built-in tools.
pub(crate) fn answers_in_plain_text(provider: &dyn Provider, session: &Session) -> bool {
    session.session_type == SessionType::SubAgent && !provider.supports_builtin_tools()
}

/// Asked while the helper's provider is still held (before the helper's session ends).
pub(crate) async fn runs_plain(services: &StateMachineServices, session: &Session) -> bool {
    match services.provider(&session.id).await {
        Ok(provider) => answers_in_plain_text(provider.as_ref(), session),
        Err(_) => false,
    }
}

/// The helper's last answer as its result, or `None` when it ended on an error or without text.
pub(crate) fn plain_output(messages: &[Message]) -> Option<String> {
    let last = messages
        .iter()
        .rev()
        .find(|message| message.role == Role::Assistant)?;
    if last.error().is_some() || last.error_kind().is_some() {
        return None;
    }
    let text = last.as_concat_text();
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    Some(serde_json::json!({ "summary": text }).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::message::MessageErrorKind;

    #[test]
    fn plain_output_takes_the_last_answer_as_summary() {
        let messages = vec![
            Message::user().with_text("read package.json"),
            Message::assistant().with_text("first look"),
            Message::user().with_text("tool result"),
            Message::assistant().with_text("  the name is goose-app  "),
        ];
        assert_eq!(
            plain_output(&messages).as_deref(),
            Some(r#"{"summary":"the name is goose-app"}"#)
        );
    }

    #[test]
    fn plain_output_is_none_without_text_or_after_an_error() {
        assert_eq!(plain_output(&[]), None);
        assert_eq!(plain_output(&[Message::assistant().with_text("  ")]), None);
        let failed = Message::assistant()
            .with_text("partial")
            .with_error(MessageErrorKind::ResourceExhausted, "turn budget exhausted");
        assert_eq!(plain_output(&[failed]), None);
    }
}
