//! Regroups parallel tool calls that were stored interleaved with their results.
//!
//! External agents (for example `claude-acp`) report parallel tool calls and
//! their results in arrival order, so a single model turn can be stored as
//! `Asst(A) → Asst(B) → Resp(A) → Asst(C) → Resp(B) → Resp(C)`. Providers with
//! strict call/result pairing (Gemini/Vertex, OpenAI, the Anthropic API) reject
//! such a history once a chat switches to them, because the calls and results
//! of one turn end up split across several turns.
//!
//! While tool calls of a turn are still unanswered, the model cannot have
//! started a new turn, so every further assistant message (tool calls, text or
//! thinking) belongs to the same turn. This step moves those assistant messages
//! in front of the tool results that were stored between them. The order of the
//! assistant messages among themselves, the order of the tool results among
//! themselves and the block order inside every message stay unchanged. A real
//! user message (not a bare tool result) always closes the turn and is never
//! moved across. `merge_consecutive_messages` then joins the now adjacent
//! messages into one assistant turn followed by one tool-result turn.
//!
//! Runs after `fix_tool_calling`, so orphaned calls and results are already
//! removed and cannot keep a turn open.

use super::message::{Message, MessageContentBlock};
use super::{effective_role, EffectiveRole};
use std::collections::HashSet;

fn tool_request_ids(message: &Message) -> impl Iterator<Item = &str> {
    message.content.iter().filter_map(|content| match content {
        MessageContentBlock::ToolRequest(req) => Some(req.id.as_str()),
        _ => None,
    })
}

fn tool_response_ids(message: &Message) -> impl Iterator<Item = &str> {
    message.content.iter().filter_map(|content| match content {
        MessageContentBlock::ToolResponse(resp) => Some(resp.id.as_str()),
        _ => None,
    })
}

pub(super) fn group_interleaved_tool_calls(messages: Vec<Message>) -> (Vec<Message>, Vec<String>) {
    let mut issues = Vec::new();
    let mut result: Vec<Message> = Vec::with_capacity(messages.len());
    let mut iter = messages.into_iter().peekable();

    while let Some(message) = iter.next() {
        let mut open_calls: HashSet<String> =
            tool_request_ids(&message).map(str::to_owned).collect();
        if effective_role(&message) != EffectiveRole::Assistant || open_calls.is_empty() {
            result.push(message);
            continue;
        }

        let mut assistant_part = vec![message];
        let mut tool_part: Vec<Message> = Vec::new();
        let mut moved = 0usize;

        while !open_calls.is_empty() {
            let Some(next) = iter.peek() else { break };
            match effective_role(next) {
                EffectiveRole::User => break,
                EffectiveRole::Assistant => {
                    let next = iter.next().expect("peeked message exists");
                    open_calls.extend(tool_request_ids(&next).map(str::to_owned));
                    if !tool_part.is_empty() {
                        moved += 1;
                    }
                    assistant_part.push(next);
                }
                EffectiveRole::Tool => {
                    let next = iter.next().expect("peeked message exists");
                    for id in tool_response_ids(&next) {
                        open_calls.remove(id);
                    }
                    tool_part.push(next);
                }
            }
        }

        if moved > 0 {
            issues.push(format!(
                "Grouped {} interleaved assistant message(s) of a parallel tool turn",
                moved
            ));
        }
        result.extend(assistant_part);
        result.extend(tool_part);
    }

    (result, issues)
}

#[cfg(test)]
mod tests {
    use super::group_interleaved_tool_calls;
    use crate::conversation::message::{Message, MessageContentBlock};
    use crate::conversation::{fix_conversation, Conversation};
    use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock, Role};
    use rmcp::object;

    fn call(id: &str) -> Message {
        Message::assistant().with_tool_request(
            id,
            Ok(CallToolRequestParams::new("search").with_arguments(object!({}))),
        )
    }

    fn calls(ids: &[&str]) -> Message {
        ids.iter().fold(Message::assistant(), |msg, id| {
            msg.with_tool_request(
                *id,
                Ok(CallToolRequestParams::new("search").with_arguments(object!({}))),
            )
        })
    }

    fn resp(ids: &[&str]) -> Message {
        ids.iter().fold(Message::user(), |msg, id| {
            msg.with_tool_response(
                *id,
                Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                    "result {id}"
                ))])),
            )
        })
    }

    /// Returns (role, block kinds with ids) per message, e.g. ("assistant", ["call:A", "text"]).
    fn shape(messages: &[Message]) -> Vec<(Role, Vec<String>)> {
        messages
            .iter()
            .map(|m| {
                let blocks = m
                    .content
                    .iter()
                    .map(|c| match c {
                        MessageContentBlock::ToolRequest(r) => format!("call:{}", r.id),
                        MessageContentBlock::ToolResponse(r) => format!("resp:{}", r.id),
                        MessageContentBlock::Text(t) => format!("text:{}", t.text),
                        MessageContentBlock::Thinking(_) => "thinking".to_string(),
                        _ => "other".to_string(),
                    })
                    .collect();
                (m.role.clone(), blocks)
            })
            .collect()
    }

    fn fixed(messages: Vec<Message>) -> Vec<Message> {
        let (fixed, _) = fix_conversation(Conversation::new_unvalidated(messages));
        let again = fix_conversation(fixed.clone());
        assert!(
            again.1.is_empty(),
            "fixed conversation must be stable, found {:?}",
            again.1
        );
        fixed.messages().clone()
    }

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|i| i.to_string()).collect()
    }

    #[test]
    fn chat_case_becomes_one_call_turn_and_one_result_turn() {
        let messages = vec![
            Message::user().with_text("go"),
            call("A"),
            call("B"),
            resp(&["A"]),
            call("C"),
            resp(&["B"]),
            resp(&["C"]),
            Message::assistant().with_text("done"),
            Message::user().with_text("next"),
        ];
        assert_eq!(
            shape(&fixed(messages)),
            vec![
                (Role::User, s(&["text:go"])),
                (Role::Assistant, s(&["call:A", "call:B", "call:C"])),
                (Role::User, s(&["resp:A", "resp:B", "resp:C"])),
                (Role::Assistant, s(&["text:done"])),
                (Role::User, s(&["text:next"])),
            ]
        );
    }

    #[test]
    fn multiple_interleavings_are_grouped() {
        let messages = vec![
            Message::user().with_text("go"),
            calls(&["A", "B"]),
            resp(&["A"]),
            call("C"),
            resp(&["B"]),
            call("D"),
            resp(&["C", "D"]),
            Message::user().with_text("next"),
        ];
        assert_eq!(
            shape(&fixed(messages)),
            vec![
                (Role::User, s(&["text:go"])),
                (
                    Role::Assistant,
                    s(&["call:A", "call:B", "call:C", "call:D"])
                ),
                (Role::User, s(&["resp:A", "resp:B", "resp:C", "resp:D"])),
                (Role::User, s(&["text:next"])),
            ]
        );
    }

    #[test]
    fn assistant_text_and_thinking_during_open_calls_do_not_split_results() {
        let messages = vec![
            Message::user().with_text("go"),
            calls(&["A", "B"]),
            resp(&["A"]),
            Message::assistant()
                .with_thinking("still waiting", "")
                .with_text("waiting"),
            resp(&["B"]),
            Message::user().with_text("next"),
        ];
        assert_eq!(
            shape(&fixed(messages)),
            vec![
                (Role::User, s(&["text:go"])),
                (
                    Role::Assistant,
                    s(&["call:A", "call:B", "thinking", "text:waiting"])
                ),
                (Role::User, s(&["resp:A", "resp:B"])),
                (Role::User, s(&["text:next"])),
            ]
        );
    }

    #[test]
    fn real_user_message_is_never_moved_across() {
        let messages = vec![
            call("A"),
            Message::user().with_text("stop"),
            call("B"),
            resp(&["A"]),
        ];
        let before = shape(&messages);
        let (after, issues) = group_interleaved_tool_calls(messages);
        assert_eq!(shape(&after), before);
        assert!(issues.is_empty());

        let messages = vec![
            calls(&["A", "B"]),
            resp(&["A"]),
            Message::user().with_text("stop"),
            call("C"),
            resp(&["B"]),
        ];
        let before = shape(&messages);
        let (after, issues) = group_interleaved_tool_calls(messages);
        assert_eq!(shape(&after), before);
        assert!(issues.is_empty());
    }

    #[test]
    fn regular_history_stays_identical() {
        let messages = vec![
            Message::user().with_text("go"),
            calls(&["A", "B"]),
            resp(&["A", "B"]),
            Message::assistant()
                .with_thinking("think", "sig")
                .with_text("next step"),
            call("C"),
            resp(&["C"]),
            Message::assistant().with_text("done"),
            Message::user().with_text("thanks"),
        ];
        let (after, issues) = group_interleaved_tool_calls(messages.clone());
        assert_eq!(after, messages);
        assert!(issues.is_empty());
    }

    #[test]
    fn mixed_blocks_keep_their_order_inside_each_message() {
        let interleaved = vec![
            Message::user().with_text("go"),
            Message::assistant()
                .with_thinking("plan", "sig")
                .with_text("first")
                .with_tool_request(
                    "A",
                    Ok(CallToolRequestParams::new("search").with_arguments(object!({}))),
                ),
            Message::assistant().with_tool_request(
                "B",
                Ok(CallToolRequestParams::new("search").with_arguments(object!({}))),
            ),
            resp(&["A"]),
            Message::assistant().with_text("third").with_tool_request(
                "C",
                Ok(CallToolRequestParams::new("search").with_arguments(object!({}))),
            ),
            resp(&["B", "C"]),
            Message::user().with_text("next"),
        ];
        assert_eq!(
            shape(&fixed(interleaved)),
            vec![
                (Role::User, s(&["text:go"])),
                (
                    Role::Assistant,
                    s(&[
                        "thinking",
                        "text:first",
                        "call:A",
                        "call:B",
                        "text:third",
                        "call:C"
                    ])
                ),
                (Role::User, s(&["resp:A", "resp:B", "resp:C"])),
                (Role::User, s(&["text:next"])),
            ]
        );
    }
}
