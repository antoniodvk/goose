//! Sends a saved Codex request once more, optionally with one variant, and
//! reports only whether the model would call a tool or stop (C214).
//!
//! The whole point is what this program does not do: it runs no tool, because it
//! holds no agent and no extension; and it never signs in or refreshes a token,
//! because it shares the sign-in with the running Goose and a rotated refresh
//! token could sign the other one out. `tokens.json` is only ever read.

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use clap::{Parser, Subcommand};
use goose::config::paths::Paths;
use goose::providers::chatgpt_codex::CODEX_API_ENDPOINT;
use goose::providers::codex_trace::{
    decision_points_dir, response_ends_path, ResponseEnd, ToolCallSeen,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// A long model turn must not outlive the token.
const MIN_TOKEN_MINUTES: i64 = 15;
/// A response this recent means another session is working on this sign-in.
const QUIET_MINUTES: i64 = 10;
const MAX_RUNS: u32 = 10;
const COMMAND_PREVIEW_CHARS: usize = 500;
const REQUEST_TIMEOUT_SECS: u64 = 600;

#[derive(Parser)]
#[command(
    about = "Spielt eine gespeicherte Codex-Anfrage erneut ab, ohne ein Werkzeug auszuführen"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Zeigt Profil, Restlaufzeit der Anmeldung und ob gerade eine Sitzung arbeitet
    Status,
    /// Listet die gespeicherten Entscheidungsstellen
    List,
    /// Spielt eine Stelle ab
    Run {
        /// Dateiname aus `list` oder Pfad, auch eine llm_request-Datei
        point: String,
        /// JSON-Datei mit den Änderungen dieser Variante
        #[arg(long)]
        variant: Option<PathBuf>,
        /// Wie oft die Stelle abgespielt wird
        #[arg(long, default_value_t = 1)]
        times: u32,
    },
}

#[derive(Debug, Deserialize)]
struct StoredToken {
    access_token: String,
    expires_at: DateTime<Utc>,
    account_id: Option<String>,
}

fn load_token() -> Result<StoredToken> {
    let path = Paths::in_config_dir("chatgpt_codex/tokens.json");
    let text = fs::read_to_string(&path).map_err(|_| {
        anyhow!(
            "Keine ChatGPT-Anmeldung in diesem Profil ({}). Dieses Werkzeug meldet sich nie an.",
            path.display()
        )
    })?;
    serde_json::from_str(&text).context("tokens.json ist nicht lesbar")
}

fn token_refusal(expires_at: DateTime<Utc>, now: DateTime<Utc>) -> Option<String> {
    let left = expires_at - now;
    (left < Duration::minutes(MIN_TOKEN_MINUTES)).then(|| {
        format!(
            "Verweigert: Die Anmeldung läuft in {} min ab. Erst muss Goose sie selbst erneuern, \
             dieses Werkzeug erneuert sie nie.",
            left.num_minutes()
        )
    })
}

fn busy_refusal(last_response: Option<SystemTime>, now: SystemTime) -> Option<String> {
    let age = now.duration_since(last_response?).ok()?;
    let minutes = age.as_secs() / 60;
    (minutes < QUIET_MINUTES as u64).then(|| {
        format!(
            "Verweigert: Die letzte Codex-Antwort in diesem Profil kam vor {minutes} min, \
             es arbeitet also vermutlich eine andere Sitzung."
        )
    })
}

fn last_response_time() -> Option<SystemTime> {
    fs::metadata(response_ends_path()).ok()?.modified().ok()
}

fn preflight(token: &StoredToken) -> Result<()> {
    if let Some(refusal) = token_refusal(token.expires_at, Utc::now()) {
        bail!(refusal);
    }
    if let Some(refusal) = busy_refusal(last_response_time(), SystemTime::now()) {
        bail!(refusal);
    }
    Ok(())
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Variant {
    #[serde(default)]
    replace: Vec<Replacement>,
    /// Dotted field path to new value, e.g. `reasoning.effort` or `model`.
    #[serde(default)]
    set: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Replacement {
    find: String,
    with: String,
}

/// Encrypted reasoning must reach the backend byte for byte.
fn replace_in_strings(value: &mut Value, find: &str, with: &str) -> usize {
    match value {
        Value::String(text) => {
            let hits = text.matches(find).count();
            if hits > 0 {
                *text = text.replace(find, with);
            }
            hits
        }
        Value::Array(items) => items
            .iter_mut()
            .map(|item| replace_in_strings(item, find, with))
            .sum(),
        Value::Object(fields) => fields
            .iter_mut()
            .filter(|(key, _)| key.as_str() != "encrypted_content")
            .map(|(_, item)| replace_in_strings(item, find, with))
            .sum(),
        _ => 0,
    }
}

fn set_field(request: &mut Value, path: &str, new_value: Value) -> Result<()> {
    let mut target = request;
    let mut parts = path.split('.').peekable();
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            target
                .as_object_mut()
                .ok_or_else(|| anyhow!("{path}: Feld liegt nicht in einem Objekt"))?
                .insert(part.to_string(), new_value);
            return Ok(());
        }
        target = target
            .as_object_mut()
            .ok_or_else(|| anyhow!("{path}: Feld liegt nicht in einem Objekt"))?
            .entry(part)
            .or_insert_with(|| json!({}));
    }
    bail!("Leerer Feldname")
}

/// Same rule as `replay_for` in Goose: reasoning and `phase` only go back to the
/// model that wrote them.
fn drop_foreign_replay(request: &mut Value) {
    if let Some(items) = request.get_mut("input").and_then(Value::as_array_mut) {
        items.retain(|item| item.get("type").and_then(Value::as_str) != Some("reasoning"));
        for item in items {
            if let Some(fields) = item.as_object_mut() {
                fields.remove("phase");
            }
        }
    }
}

/// A replacement that hits nothing would silently replay the control run.
fn apply_variant(request: &mut Value, variant: &Variant) -> Result<()> {
    for replacement in &variant.replace {
        let mut hits = 0;
        for field in ["instructions", "input"] {
            if let Some(value) = request.get_mut(field) {
                hits += replace_in_strings(value, &replacement.find, &replacement.with);
            }
        }
        if hits == 0 {
            bail!("Variante trifft nichts: {:?}", replacement.find);
        }
    }
    let old_model = request.get("model").cloned();
    for (path, new_value) in &variant.set {
        set_field(request, path, new_value.clone())?;
    }
    if request.get("model").cloned() != old_model {
        drop_foreign_replay(request);
    }
    Ok(())
}

/// The decision depends on the stream ending normally and on nothing being
/// stored at OpenAI, whatever a variant says.
fn fix_transport(request: &mut Value) -> Result<()> {
    let fields = request
        .as_object_mut()
        .ok_or_else(|| anyhow!("Die Anfrage ist kein JSON-Objekt"))?;
    fields.insert("stream".to_string(), json!(true));
    fields.insert("store".to_string(), json!(false));
    Ok(())
}

/// A decision point holds `request`, a `llm_request` log starts with `input`.
fn request_from_text(text: &str) -> Result<Value> {
    let first = text.lines().next().unwrap_or_default();
    let stored: Value = serde_json::from_str(text)
        .or_else(|_| serde_json::from_str(first))
        .context("Die Stelle ist kein JSON")?;
    ["request", "input"]
        .iter()
        .find_map(|key| stored.get(*key).filter(|value| value.is_object()).cloned())
        .ok_or_else(|| anyhow!("Die Datei enthält keine gespeicherte Anfrage"))
}

fn point_path(point: &str) -> PathBuf {
    let path = Path::new(point);
    if path.exists() {
        path.to_path_buf()
    } else {
        decision_points_dir().join(point)
    }
}

fn load_request(point: &str) -> Result<Value> {
    let path = point_path(point);
    let text = fs::read_to_string(&path)
        .with_context(|| format!("Stelle nicht gefunden: {}", path.display()))?;
    request_from_text(&text)
}

fn load_variant(path: Option<&Path>) -> Result<Variant> {
    let Some(path) = path else {
        return Ok(Variant::default());
    };
    let text = fs::read_to_string(path)
        .with_context(|| format!("Variante nicht gefunden: {}", path.display()))?;
    serde_json::from_str(&text).context("Variante ist ungültig")
}

#[derive(Debug, PartialEq)]
enum Replayed {
    Answered(ResponseEnd),
    NotTestable { http_status: u16 },
}

/// Reads until the response is finished; it never answers a tool call.
async fn replay_once(
    client: &reqwest::Client,
    url: &str,
    token: &StoredToken,
    request: &Value,
) -> Result<Replayed> {
    let mut call = client
        .post(url)
        .bearer_auth(&token.access_token)
        .json(request);
    if let Some(account_id) = &token.account_id {
        call = call.header("chatgpt-account-id", account_id);
    }
    let mut response = call.send().await?;
    if !response.status().is_success() {
        return Ok(Replayed::NotTestable {
            http_status: response.status().as_u16(),
        });
    }
    let mut end = ResponseEnd::default();
    let mut pending = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        pending.extend_from_slice(&chunk);
        while let Some(line_end) = pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = pending.drain(..=line_end).collect();
            end.observe(String::from_utf8_lossy(&line).trim_end());
        }
        if end.is_finished() {
            break;
        }
    }
    Ok(Replayed::Answered(end))
}

fn command_of(call: &ToolCallSeen) -> String {
    let arguments = serde_json::from_str::<Value>(&call.arguments).ok();
    let command = arguments
        .as_ref()
        .and_then(|value| value.get("command"))
        .and_then(Value::as_str)
        .unwrap_or(&call.arguments);
    command.chars().take(COMMAND_PREVIEW_CHARS).collect()
}

fn verdict(end: &ResponseEnd) -> &'static str {
    if !end.is_finished() {
        "nicht prüfbar"
    } else if end.calls_tool() {
        "Werkzeugaufruf"
    } else {
        "Schluss"
    }
}

fn result_line(run: u32, end: &ResponseEnd) -> Value {
    let tools: Vec<Value> = end
        .tool_calls
        .iter()
        .map(|call| json!({ "name": call.name, "befehl": command_of(call) }))
        .collect();
    json!({
        "lauf": run,
        "ergebnis": verdict(end),
        "status": end.status,
        "grund": end.incomplete_reason,
        "end_turn": end.end_turn,
        "phasen": end.phases,
        "werkzeuge": tools,
        "antwortfelder": end.response_fields,
        "eingabe_tokens": end.input_tokens,
        "ausgabe_tokens": end.output_tokens,
    })
}

async fn run(point: &str, variant: Option<&Path>, times: u32) -> Result<()> {
    if times == 0 || times > MAX_RUNS {
        bail!("--times muss zwischen 1 und {MAX_RUNS} liegen");
    }
    let mut request = load_request(point)?;
    apply_variant(&mut request, &load_variant(variant)?)?;
    fix_transport(&mut request)?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .build()?;
    let url = format!("{CODEX_API_ENDPOINT}/responses");
    for run in 1..=times {
        // Read anew each time: a running Goose may have rotated the token.
        let token = load_token()?;
        preflight(&token)?;
        match replay_once(&client, &url, &token, &request).await? {
            Replayed::Answered(end) => println!("{}", result_line(run, &end)),
            Replayed::NotTestable { http_status } => {
                println!(
                    "{}",
                    json!({ "lauf": run, "ergebnis": "nicht prüfbar", "http": http_status })
                );
                bail!("Abbruch: HTTP {http_status}. Dieses Werkzeug meldet sich nie an.");
            }
        }
    }
    Ok(())
}

fn status() -> Result<()> {
    println!("Profil: {}", Paths::config_dir().display());
    match load_token() {
        Ok(token) => {
            let left = token.expires_at - Utc::now();
            println!("Anmeldung: gültig für noch {} min", left.num_minutes());
            if let Some(refusal) = token_refusal(token.expires_at, Utc::now()) {
                println!("{refusal}");
            }
        }
        Err(error) => println!("{error}"),
    }
    match busy_refusal(last_response_time(), SystemTime::now()) {
        Some(refusal) => println!("{refusal}"),
        None => println!("Keine Codex-Antwort in den letzten {QUIET_MINUTES} min."),
    }
    Ok(())
}

fn list() -> Result<()> {
    let dir = decision_points_dir();
    let Ok(entries) = fs::read_dir(&dir) else {
        println!("Noch keine Entscheidungsstellen in {}", dir.display());
        return Ok(());
    };
    let mut points: Vec<_> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
        .collect();
    points.sort();
    for (_, path) in points {
        let end = fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .map(|point| point["end"].clone())
            .unwrap_or_default();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        println!(
            "{name}  {}  {}  {}  end_turn={}  phasen={}  werkzeuge={}",
            end["ts"], end["model"], end["outcome"], end["end_turn"], end["phases"], end["tools"]
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Status => status(),
        Command::List => list(),
        Command::Run {
            point,
            variant,
            times,
        } => run(&point, variant.as_deref(), times).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration as StdDuration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn variant(text: Value) -> Variant {
        serde_json::from_value(text).unwrap()
    }

    fn stored_request() -> Value {
        json!({
            "model": "gpt-6.1-terra",
            "instructions": "Rule A. Rule A.",
            "store": true,
            "input": [
                { "type": "reasoning", "encrypted_content": "Rule A" },
                { "role": "assistant", "phase": "commentary", "content": [
                    { "type": "output_text", "text": "Rule A" },
                ]},
                { "type": "function_call_output", "call_id": "c", "output": "no match" },
            ],
            "tools": [{ "name": "shell", "description": "Rule A" }],
        })
    }

    fn token() -> StoredToken {
        StoredToken {
            access_token: "access".to_string(),
            expires_at: Utc::now() + Duration::hours(1),
            account_id: Some("account".to_string()),
        }
    }

    /// Answers every connection with `response` and counts the requests.
    async fn fake_backend(response: String) -> (String, tokio::task::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/responses", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let mut seen = Vec::new();
            while let Ok(Ok((mut socket, _))) =
                tokio::time::timeout(StdDuration::from_millis(300), listener.accept()).await
            {
                let mut buffer = vec![0; 16 * 1024];
                let size = socket.read(&mut buffer).await.unwrap();
                seen.push(String::from_utf8_lossy(&buffer[..size]).to_lowercase());
                socket.write_all(response.as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            }
            seen
        });
        (url, server)
    }

    fn sse(body: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\
             connection: close\r\n\r\n{body}",
            body.len()
        )
    }

    #[test]
    fn a_replacement_reaches_rules_and_messages_but_not_encrypted_reasoning() {
        let mut request = stored_request();

        apply_variant(
            &mut request,
            &variant(json!({ "replace": [{ "find": "Rule A", "with": "Rule B" }] })),
        )
        .unwrap();

        assert_eq!(request["instructions"], "Rule B. Rule B.");
        assert_eq!(request["input"][1]["content"][0]["text"], "Rule B");
        assert_eq!(request["input"][0]["encrypted_content"], "Rule A");
        assert_eq!(request["tools"][0]["description"], "Rule A");
    }

    #[test]
    fn a_replacement_without_a_hit_is_refused() {
        let mut request = stored_request();

        let error = apply_variant(
            &mut request,
            &variant(json!({ "replace": [{ "find": "missing", "with": "x" }] })),
        )
        .unwrap_err();

        assert!(error.to_string().contains("trifft nichts"));
    }

    #[test]
    fn a_higher_reasoning_effort_leaves_the_replay_in_place() {
        let mut request = stored_request();

        apply_variant(
            &mut request,
            &variant(json!({ "set": { "reasoning.effort": "high" } })),
        )
        .unwrap();

        assert_eq!(request["reasoning"]["effort"], "high");
        assert_eq!(request["input"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn another_model_gets_no_foreign_reasoning_or_phase() {
        let mut request = stored_request();

        apply_variant(
            &mut request,
            &variant(json!({ "set": { "model": "gpt-6-sol" } })),
        )
        .unwrap();

        let input = request["input"].as_array().unwrap();
        assert_eq!(input.len(), 2);
        assert!(input[0].get("phase").is_none());
    }

    #[test]
    fn unknown_variant_fields_are_refused() {
        let result: Result<Variant, _> = serde_json::from_value(json!({ "replce": [] }));

        assert!(result.is_err());
    }

    #[test]
    fn transport_fields_cannot_be_changed_by_a_variant() {
        let mut request = stored_request();
        apply_variant(
            &mut request,
            &variant(json!({ "set": { "stream": false } })),
        )
        .unwrap();

        fix_transport(&mut request).unwrap();

        assert_eq!(request["stream"], true);
        assert_eq!(request["store"], false);
    }

    #[test]
    fn a_token_about_to_expire_is_refused() {
        let now = Utc::now();

        assert!(token_refusal(now + Duration::minutes(14), now).is_some());
        assert!(token_refusal(now - Duration::minutes(1), now).is_some());
        assert!(token_refusal(now + Duration::minutes(16), now).is_none());
    }

    #[test]
    fn a_recent_response_blocks_the_replay() {
        let now = SystemTime::now();

        assert!(busy_refusal(Some(now - StdDuration::from_secs(120)), now).is_some());
        assert!(busy_refusal(Some(now - StdDuration::from_secs(11 * 60)), now).is_none());
        assert!(busy_refusal(None, now).is_none());
    }

    #[test]
    fn a_decision_point_and_a_request_log_both_load() {
        let point = json!({ "end": {}, "request": { "model": "m" } }).to_string();
        let log = format!(
            "{}\n{}\n",
            json!({ "model_config": {}, "input": { "model": "m" } }),
            json!({ "data": {}, "usage": null }),
        );

        assert_eq!(request_from_text(&point).unwrap()["model"], "m");
        assert_eq!(request_from_text(&log).unwrap()["model"], "m");
        assert!(request_from_text("{}").is_err());
    }

    #[test]
    fn the_command_is_shown_for_a_handover() {
        let call = ToolCallSeen {
            name: "developer__shell".to_string(),
            arguments: json!({ "command": "pwsh cc-sitzung-wechseln.ps1" }).to_string(),
        };

        assert_eq!(command_of(&call), "pwsh cc-sitzung-wechseln.ps1");
    }

    #[test]
    fn a_tool_without_a_command_shows_its_arguments_cut_short() {
        let call = ToolCallSeen {
            name: "read".to_string(),
            arguments: "x".repeat(COMMAND_PREVIEW_CHARS + 1),
        };

        assert_eq!(command_of(&call).len(), COMMAND_PREVIEW_CHARS);
    }

    #[tokio::test]
    async fn a_tool_call_is_reported_and_answered_with_nothing() {
        let body = concat!(
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",",
            "\"call_id\":\"c\",\"name\":\"developer__shell\",\"arguments\":\"{}\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",",
            "\"status\":\"completed\",\"end_turn\":false}}\n\n",
        );
        let (url, server) = fake_backend(sse(body)).await;

        let result = replay_once(&reqwest::Client::new(), &url, &token(), &json!({}))
            .await
            .unwrap();

        let Replayed::Answered(end) = result else {
            panic!("expected an answer");
        };
        assert_eq!(verdict(&end), "Werkzeugaufruf");
        assert_eq!(end.end_turn, Some(false));
        let seen = server.await.unwrap();
        assert_eq!(
            seen.len(),
            1,
            "a tool call must not lead to a second request"
        );
        assert!(seen[0].contains("authorization: bearer access"));
        assert!(seen[0].contains("chatgpt-account-id: account"));
    }

    #[tokio::test]
    async fn a_text_answer_counts_as_stop() {
        let body = concat!(
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",",
            "\"phase\":\"final_answer\",\"content\":[]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",",
            "\"status\":\"completed\"}}\n\n",
        );
        let (url, server) = fake_backend(sse(body)).await;

        let result = replay_once(&reqwest::Client::new(), &url, &token(), &json!({}))
            .await
            .unwrap();
        server.await.unwrap();

        let Replayed::Answered(end) = result else {
            panic!("expected an answer");
        };
        assert_eq!(verdict(&end), "Schluss");
    }

    #[tokio::test]
    async fn a_rejected_token_is_not_testable_and_never_retried() {
        let response =
            "HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\nconnection: close\r\n\r\n";
        let (url, server) = fake_backend(response.to_string()).await;

        let result = replay_once(&reqwest::Client::new(), &url, &token(), &json!({}))
            .await
            .unwrap();

        assert_eq!(result, Replayed::NotTestable { http_status: 401 });
        assert_eq!(server.await.unwrap().len(), 1);
    }

    #[test]
    fn a_stream_that_ends_early_is_not_a_stop() {
        assert_eq!(verdict(&ResponseEnd::default()), "nicht prüfbar");
    }
}
