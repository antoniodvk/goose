//! Deferred extension tools and `tool_search` (C161).

use anyhow::Result;
use serde_json::json;

use super::calculator_extension::{value, ADD, DIVIDE};
use super::pipeline::{test_pipeline, MessageKind::Agent, MessageKind::ToolResponse};
use crate::agents::tool_search::TOOL_SEARCH_TOOL_NAME;
use crate::config::GooseMode;

#[tokio::test]
async fn without_tool_search_every_extension_tool_is_advertised() -> Result<()> {
    let (pipeline, api) = test_pipeline().await?;
    api.on("hello").reply("hi");

    pipeline.run(["hello"]).await?;

    let call = &api.calls()[0];
    assert!(call.advertises_tool(ADD));
    assert!(!call.advertises_tool(TOOL_SEARCH_TOOL_NAME));
    assert!(!call.system_contains("# Deferred tools"));
    Ok(())
}

#[tokio::test]
async fn tool_search_loads_a_deferred_tool_for_the_rest_of_the_conversation() -> Result<()> {
    let (pipeline, api) = test_pipeline().await?;
    let pipeline = pipeline.with_tool_search();
    api.on("add one").call(
        TOOL_SEARCH_TOOL_NAME,
        json!({ "query": "select:calculator__add" }),
    );
    api.on("Loaded; these tools").call(ADD, value(1));
    api.on("result: 1").reply("added");

    let result = pipeline.run(["add one"]).await?;

    let calls = api.calls();
    assert!(calls[0].system_contains("# Deferred tools"));
    assert!(calls[0].system_contains(ADD));
    assert!(calls[0].system_contains("listed under Deferred tools"));
    assert!(calls[0].advertises_tool(TOOL_SEARCH_TOOL_NAME));
    assert!(
        calls[0].advertises_tool("todo__todo_write"),
        "goose's own tools stay"
    );
    assert!(!calls[0].advertises_tool(ADD));
    assert!(calls[1].advertises_tool(ADD));
    assert!(
        !calls[1].advertises_tool(DIVIDE),
        "only the selected tool loads"
    );
    assert_eq!(
        calls[0].tool_schema(TOOL_SEARCH_TOOL_NAME),
        calls[1].tool_schema(TOOL_SEARCH_TOOL_NAME)
    );
    assert_eq!(pipeline.calculator_total(), 1);
    result.assert_message(-4, ToolResponse, "- calculator__add: ");
    result.assert_message(-1, Agent, "added");

    api.on("next turn").reply("still here");
    pipeline.run(["next turn"]).await?;
    let last = api.calls().pop().unwrap();
    assert!(last.advertises_tool(ADD), "a loaded tool stays loaded");
    assert!(last.system_contains("# Deferred tools"));
    Ok(())
}

/// A direct call is refused like an unknown tool, with a hint to tool_search; since the
/// conversation now names the tool, it is in the list from the next request on.
#[tokio::test]
async fn calling_a_deferred_tool_directly_is_refused_once() -> Result<()> {
    let (pipeline, api) = test_pipeline().await?;
    let pipeline = pipeline.with_tool_search();
    api.on("add two").unadvertised_call(ADD, value(2));
    api.on("loaded with tool_search first").call(ADD, value(2));
    api.on("result: 2").reply("added");

    let result = pipeline.run(["add two"]).await?;

    let calls = api.calls();
    assert!(!calls[0].advertises_tool(ADD));
    assert!(calls[1].input_contains("is not available"));
    assert!(calls[1].advertises_tool(ADD));
    assert_eq!(pipeline.calculator_total(), 2, "only the second call ran");
    result.assert_message(-1, Agent, "added");
    Ok(())
}

#[tokio::test]
async fn keyword_search_loads_the_best_matches_without_approval() -> Result<()> {
    let (pipeline, api) = test_pipeline().await?;
    let pipeline = pipeline
        .with_tool_search()
        .with_goose_mode(GooseMode::Approve)
        .await;
    api.on("find a divider").call(
        TOOL_SEARCH_TOOL_NAME,
        json!({ "query": "+calculator divide", "max_results": 1 }),
    );
    api.on("Loaded; these tools").reply("ready");

    let result = pipeline.run(["find a divider"]).await?;

    let calls = api.calls();
    assert_eq!(
        calls.len(),
        2,
        "tool_search ran without an approval request"
    );
    assert!(calls[1].advertises_tool(DIVIDE));
    assert!(!calls[1].advertises_tool(ADD));
    result.assert_message(-2, ToolResponse, "calculator__divide");
    Ok(())
}
