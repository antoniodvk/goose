//! Account usage of a provider: plan windows (how much of an allowance is used
//! and when it resets) and balances, as the provider's own account API reports
//! them. Not to be confused with `conversation::token_usage`, which counts the
//! tokens of one request.
//!
//! The parsers take the JSON a provider answers with and keep the provider's
//! own window names in `UsageWindow::id`; nothing here knows how a window is
//! displayed. Field names are quoted from the primary sources named at each
//! parser.

use crate::errors::ProviderError;
use chrono::DateTime;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageCurrency {
    Usd,
    Credits,
}

/// One allowance window of a plan.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UsageWindow {
    /// The provider's own name for the window (for example `primary`, `dailyInputTokens`).
    pub id: String,
    /// 0 to 100 (may exceed 100 when the provider allows overage); `None` when the provider reports "unknown".
    pub used_percent: Option<f64>,
    /// Unix time in milliseconds.
    pub resets_at: Option<i64>,
    pub window_seconds: Option<i64>,
    pub used: Option<f64>,
    pub limit: Option<f64>,
    /// `tokens`, `images`, `usd`, `credits`.
    pub unit: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UsageBalance {
    pub id: String,
    pub amount: f64,
    pub currency: UsageCurrency,
    /// What the balance started from, when the provider reports it.
    pub granted: Option<f64>,
    /// Unix time in milliseconds.
    pub expires_at: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct AccountUsage {
    pub plan: Option<String>,
    pub windows: Vec<UsageWindow>,
    pub balances: Vec<UsageBalance>,
}

fn unexpected(what: &str) -> ProviderError {
    ProviderError::RequestFailed(format!("Unexpected answer from the usage endpoint: {what}"))
}

/// A JSON number, or a string holding one (some providers send amounts as strings).
fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok().filter(|n| n.is_finite()),
        _ => None,
    }
}

fn rfc3339_millis(value: &Value) -> Option<i64> {
    DateTime::parse_from_rfc3339(value.as_str()?)
        .ok()
        .map(|time| time.timestamp_millis())
}

/// NanoGPT: `GET /api/subscription/v1/usage` and `POST /api/check-balance`.
///
/// Source: https://docs.nano-gpt.com/api-reference/endpoint/subscription-usage.md
/// and https://docs.nano-gpt.com/api-reference/management-api.md (example
/// response, field definitions) and
/// https://docs.nano-gpt.com/api-reference/endpoint/check-balance.md.
/// `percentUsed` is a fraction (multiply by 100) and may exceed 1; `resetAt`
/// is Unix milliseconds; counters that are null with `degraded: true` mean
/// "unknown". Without an active subscription only the balance is reported.
pub fn parse_nanogpt_usage(
    subscription: &Value,
    balance: &Value,
) -> Result<AccountUsage, ProviderError> {
    let mut usage = AccountUsage::default();

    if subscription.get("active").and_then(Value::as_bool) == Some(true) {
        // (window name, unit, where its limit is stated)
        let quotas: [(&str, &str, Option<&Value>); 4] = [
            (
                "dailyInputTokens",
                "tokens",
                subscription.pointer("/limits/dailyInputTokens"),
            ),
            (
                "weeklyInputTokens",
                "tokens",
                subscription.pointer("/limits/weeklyInputTokens"),
            ),
            (
                "dailyImages",
                "images",
                subscription.pointer("/limits/dailyImages"),
            ),
            (
                "tokens",
                "tokens",
                subscription.pointer("/tokenLimits/total"),
            ),
        ];
        for (id, unit, limit) in quotas {
            // Null: this quota is not configured or not applicable.
            let Some(quota) = subscription.get(id).filter(|value| value.is_object()) else {
                continue;
            };
            let degraded = quota.get("degraded").and_then(Value::as_bool) == Some(true);
            let counter = |key: &str| {
                if degraded {
                    None
                } else {
                    quota.get(key).and_then(number)
                }
            };
            usage.windows.push(UsageWindow {
                id: id.to_string(),
                used_percent: counter("percentUsed").map(|fraction| fraction * 100.0),
                resets_at: quota.get("resetAt").and_then(number).map(|ms| ms as i64),
                window_seconds: None,
                used: counter("used"),
                limit: limit.and_then(number),
                unit: Some(unit.to_string()),
            });
        }
    }

    let amount = balance
        .get("usd_balance")
        .and_then(number)
        .ok_or_else(|| unexpected("check-balance has no usd_balance"))?;
    usage.balances.push(UsageBalance {
        id: "usd".to_string(),
        amount,
        currency: UsageCurrency::Usd,
        granted: None,
        expires_at: None,
    });
    Ok(usage)
}

/// ChatGPT: `GET https://chatgpt.com/backend-api/wham/usage`.
///
/// Source: openai/codex `codex-rs/backend-client/src/client/rate_limit_resets.rs`
/// (`rate_limit_status_url`, `/wham/usage`) and the generated models in
/// `codex-rs/codex-backend-openapi-models/src/models/`:
/// `rate_limit_status_payload.rs` (`plan_type`, `rate_limit`, `credits`,
/// `additional_rate_limits`), `rate_limit_status_details.rs` (`primary_window`,
/// `secondary_window`), `rate_limit_window_snapshot.rs` (`used_percent`,
/// `limit_window_seconds`, `reset_after_seconds`, `reset_at` in seconds),
/// `credit_status_details.rs` (`has_credits`, `unlimited`, `balance` as a
/// string). The windows are reported as they come; which ones a plan has is
/// the server's decision.
pub fn parse_chatgpt_usage(payload: &Value) -> Result<AccountUsage, ProviderError> {
    let plan = payload
        .get("plan_type")
        .and_then(Value::as_str)
        .ok_or_else(|| unexpected("wham/usage has no plan_type"))?;
    let mut usage = AccountUsage {
        plan: Some(plan.to_string()),
        ..Default::default()
    };

    let mut push_windows = |prefix: &str, details: Option<&Value>| {
        for name in ["primary_window", "secondary_window"] {
            let Some(window) = details
                .and_then(|details| details.get(name))
                .filter(|window| window.is_object())
            else {
                continue;
            };
            let id = name.trim_end_matches("_window");
            usage.windows.push(UsageWindow {
                id: format!("{prefix}{id}"),
                used_percent: window.get("used_percent").and_then(number),
                resets_at: window
                    .get("reset_at")
                    .and_then(number)
                    .map(|seconds| seconds as i64 * 1000),
                window_seconds: window
                    .get("limit_window_seconds")
                    .and_then(number)
                    .map(|seconds| seconds as i64)
                    .filter(|seconds| *seconds > 0),
                used: None,
                limit: None,
                unit: None,
            });
        }
    };
    push_windows("", payload.get("rate_limit"));
    if let Some(additional) = payload
        .get("additional_rate_limits")
        .and_then(Value::as_array)
    {
        for entry in additional {
            if let Some(name) = entry.get("limit_name").and_then(Value::as_str) {
                push_windows(&format!("{name}:"), entry.get("rate_limit"));
            }
        }
    }

    // `unlimited` credits have no amount to show; the unit of `balance` is not
    // stated in the sources, ChatGPT sells them as credits.
    if let Some(credits) = payload.get("credits").filter(|value| value.is_object()) {
        let unlimited = credits.get("unlimited").and_then(Value::as_bool) == Some(true);
        if let (false, Some(amount)) = (unlimited, credits.get("balance").and_then(number)) {
            usage.balances.push(UsageBalance {
                id: "credits".to_string(),
                amount,
                currency: UsageCurrency::Credits,
                granted: None,
                expires_at: None,
            });
        }
    }
    Ok(usage)
}

/// Grok: `GET https://cli-chat-proxy.grok.com/v1/billing?format=credits`.
///
/// Source: xai-org/grok-build `crates/codegen/xai-grok-shell/src/extensions/billing.rs`
/// (`BillingConfigResponse { config }`, `BillingConfig` with
/// `creditUsagePercent` 0.0 to 100.0, `currentPeriod { type, start, end }` as
/// RFC 3339, `prepaidBalance { val }` in USD cents; test
/// `billing_config_deserializes_credits_config_shape` for a sample answer).
pub fn parse_grok_billing(payload: &Value) -> Result<AccountUsage, ProviderError> {
    let config = payload
        .get("config")
        .filter(|config| config.is_object())
        .ok_or_else(|| unexpected("billing has no config"))?;
    let mut usage = AccountUsage::default();

    if let Some(percent) = config.get("creditUsagePercent").and_then(number) {
        let period = config.get("currentPeriod");
        let start = period.and_then(|p| p.get("start")).and_then(rfc3339_millis);
        let end = period.and_then(|p| p.get("end")).and_then(rfc3339_millis);
        usage.windows.push(UsageWindow {
            id: period
                .and_then(|p| p.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("credits")
                .to_string(),
            used_percent: Some(percent),
            resets_at: end,
            window_seconds: start
                .zip(end)
                .map(|(start, end)| (end - start) / 1000)
                .filter(|seconds| *seconds > 0),
            used: None,
            limit: None,
            unit: None,
        });
    }

    // proto3 JSON leaves out zero values, so `{}` is a zero Cent.
    if let Some(prepaid) = config.get("prepaidBalance").filter(|v| v.is_object()) {
        let cents = prepaid.get("val").and_then(number).unwrap_or(0.0);
        usage.balances.push(UsageBalance {
            id: "prepaid".to_string(),
            amount: cents / 100.0,
            currency: UsageCurrency::Usd,
            granted: None,
            expires_at: None,
        });
    }
    Ok(usage)
}

/// OpenRouter: `GET https://openrouter.ai/api/v1/key`.
///
/// Source: https://openrouter.ai/docs/api/reference/limits.md (type `Key`):
/// `limit` (credit limit of the key or null), `limit_remaining`, `is_free_tier`
/// ("whether the user has paid for credits before"), `budgets[]` with
/// `scope`, `resets_at` (ISO 8601 or null). Only a key with a limit gets a
/// window and a balance; the account balance needs a management key and is not
/// read.
pub fn parse_openrouter_key(payload: &Value) -> Result<AccountUsage, ProviderError> {
    let data = payload
        .get("data")
        .filter(|data| data.is_object())
        .ok_or_else(|| unexpected("key has no data"))?;
    let mut usage = AccountUsage {
        plan: data
            .get("is_free_tier")
            .and_then(Value::as_bool)
            .map(|free| if free { "free" } else { "paid" }.to_string()),
        ..Default::default()
    };

    let limit = data.get("limit").and_then(number);
    let remaining = data.get("limit_remaining").and_then(number);
    if let (Some(limit), Some(remaining)) = (limit, remaining) {
        let used = limit - remaining;
        let resets_at = data
            .get("budgets")
            .and_then(Value::as_array)
            .and_then(|budgets| {
                budgets
                    .iter()
                    .find(|budget| budget.get("scope").and_then(Value::as_str) == Some("key_limit"))
            })
            .and_then(|budget| budget.get("resets_at"))
            .and_then(rfc3339_millis);
        usage.windows.push(UsageWindow {
            id: "keyLimit".to_string(),
            used_percent: (limit > 0.0).then(|| used / limit * 100.0),
            resets_at,
            window_seconds: None,
            used: Some(used),
            limit: Some(limit),
            unit: Some("usd".to_string()),
        });
        usage.balances.push(UsageBalance {
            id: "keyLimit".to_string(),
            amount: remaining,
            currency: UsageCurrency::Usd,
            granted: Some(limit),
            expires_at: None,
        });
    }
    Ok(usage)
}

/// Experiential Labs: `GET https://api.experientiallabs.ai/api/v1/credits`.
///
/// Source: https://platform.experientiallabs.ai/docs/cost-api (section
/// "Credit balance & activity") and https://platform.experientiallabs.ai/llms.txt
/// (Cost API, "Balance"): `{data:{total_credits, total_usage}}` in USD, the
/// remaining balance is `total_credits - total_usage`.
pub fn parse_experiential_credits(payload: &Value) -> Result<AccountUsage, ProviderError> {
    let data = payload
        .get("data")
        .ok_or_else(|| unexpected("credits has no data"))?;
    let total = data
        .get("total_credits")
        .and_then(number)
        .ok_or_else(|| unexpected("credits has no total_credits"))?;
    let used = data
        .get("total_usage")
        .and_then(number)
        .ok_or_else(|| unexpected("credits has no total_usage"))?;
    Ok(AccountUsage {
        plan: None,
        windows: Vec::new(),
        balances: vec![UsageBalance {
            id: "credits".to_string(),
            amount: total - used,
            currency: UsageCurrency::Usd,
            granted: Some(total),
            expires_at: None,
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn nanogpt_subscription_and_balance() {
        // Example response of the management-api page, plus the check-balance example.
        let subscription = json!({
            "active": true,
            "state": "active",
            "limits": { "dailyInputTokens": 1000000, "weeklyInputTokens": 6000000, "dailyImages": 10 },
            "dailyInputTokens": { "used": 250000, "remaining": 750000, "percentUsed": 0.25, "resetAt": 1788652800000_i64 },
            "weeklyInputTokens": { "used": 1500000, "remaining": 4500000, "percentUsed": 0.25, "resetAt": 1788739200000_i64 },
            "dailyImages": { "used": 2, "remaining": 8, "percentUsed": 0.2, "resetAt": 1788652800000_i64 },
            "period": { "currentPeriodEnd": "2026-10-05T00:00:00.000Z" }
        });
        let balance = json!({ "usd_balance": "129.46956147", "nano_balance": "26.71801147" });

        let usage = parse_nanogpt_usage(&subscription, &balance).unwrap();

        assert_eq!(usage.plan, None);
        let ids: Vec<_> = usage.windows.iter().map(|w| w.id.as_str()).collect();
        assert_eq!(
            ids,
            ["dailyInputTokens", "weeklyInputTokens", "dailyImages"]
        );
        let daily = &usage.windows[0];
        assert_eq!(daily.used_percent, Some(25.0));
        assert_eq!(daily.resets_at, Some(1788652800000));
        assert_eq!(daily.used, Some(250000.0));
        assert_eq!(daily.limit, Some(1000000.0));
        assert_eq!(daily.unit.as_deref(), Some("tokens"));
        assert_eq!(usage.windows[2].unit.as_deref(), Some("images"));
        assert_eq!(usage.balances.len(), 1);
        assert_eq!(usage.balances[0].amount, 129.46956147);
        assert_eq!(usage.balances[0].currency, UsageCurrency::Usd);
    }

    #[test]
    fn nanogpt_degraded_quota_is_unknown_not_zero() {
        let subscription = json!({
            "active": true,
            "state": "active",
            "limits": { "dailyInputTokens": 1000000, "weeklyInputTokens": null, "dailyImages": null },
            "dailyInputTokens": { "used": null, "remaining": null, "percentUsed": null, "resetAt": null, "degraded": true },
            "weeklyInputTokens": null,
            "dailyImages": null,
            "period": { "currentPeriodEnd": null }
        });
        let usage = parse_nanogpt_usage(&subscription, &json!({ "usd_balance": "1.5" })).unwrap();

        assert_eq!(usage.windows.len(), 1);
        assert_eq!(usage.windows[0].used_percent, None);
        assert_eq!(usage.windows[0].used, None);
        assert_eq!(usage.windows[0].limit, Some(1000000.0));
    }

    #[test]
    fn nanogpt_without_subscription_reports_only_the_balance() {
        let subscription = json!({ "active": false, "state": "inactive" });
        let usage = parse_nanogpt_usage(&subscription, &json!({ "usd_balance": "2" })).unwrap();

        assert!(usage.windows.is_empty());
        assert_eq!(usage.balances[0].amount, 2.0);
    }

    #[test]
    fn nanogpt_balance_without_usd_is_an_error() {
        assert!(parse_nanogpt_usage(&json!({ "active": false }), &json!({})).is_err());
    }

    #[test]
    fn chatgpt_windows_and_credits() {
        // Wire names from the generated codex-backend models.
        let payload = json!({
            "plan_type": "plus",
            "rate_limit": {
                "allowed": true,
                "limit_reached": false,
                "primary_window": { "used_percent": 42, "limit_window_seconds": 18000, "reset_after_seconds": 100, "reset_at": 1788652800 },
                "secondary_window": { "used_percent": 84, "limit_window_seconds": 604800, "reset_after_seconds": 100, "reset_at": 1789000000 }
            },
            "additional_rate_limits": [{
                "limit_name": "codex_other",
                "metered_feature": "codex_other",
                "rate_limit": {
                    "allowed": true,
                    "limit_reached": false,
                    "primary_window": { "used_percent": 70, "limit_window_seconds": 900, "reset_after_seconds": 0, "reset_at": 789 }
                }
            }],
            "credits": { "has_credits": true, "unlimited": false, "balance": "9.99" }
        });

        let usage = parse_chatgpt_usage(&payload).unwrap();

        assert_eq!(usage.plan.as_deref(), Some("plus"));
        let ids: Vec<_> = usage.windows.iter().map(|w| w.id.as_str()).collect();
        assert_eq!(ids, ["primary", "secondary", "codex_other:primary"]);
        assert_eq!(usage.windows[0].used_percent, Some(42.0));
        assert_eq!(usage.windows[0].window_seconds, Some(18000));
        assert_eq!(usage.windows[0].resets_at, Some(1788652800000));
        assert_eq!(usage.balances[0].amount, 9.99);
        assert_eq!(usage.balances[0].currency, UsageCurrency::Credits);
    }

    #[test]
    fn chatgpt_without_rate_limit_or_credit_balance() {
        let payload = json!({
            "plan_type": "free",
            "rate_limit": null,
            "credits": { "has_credits": false, "unlimited": true, "balance": null }
        });

        let usage = parse_chatgpt_usage(&payload).unwrap();

        assert!(usage.windows.is_empty());
        assert!(usage.balances.is_empty());
    }

    #[test]
    fn chatgpt_answer_without_plan_is_an_error() {
        assert!(parse_chatgpt_usage(&json!({ "rate_limit": null })).is_err());
    }

    #[test]
    fn grok_credits_config() {
        // Sample answer from grok-build's test `billing_config_deserializes_credits_config_shape`.
        let payload = json!({
            "config": {
                "creditUsagePercent": 42.5,
                "currentPeriod": {
                    "type": "USAGE_PERIOD_TYPE_WEEKLY",
                    "start": "2026-06-01T00:00:00Z",
                    "end": "2026-06-08T00:00:00Z"
                },
                "onDemandCap": { "val": 5000 },
                "onDemandUsed": { "val": 300 },
                "prepaidBalance": { "val": 1250 },
                "isUnifiedBillingUser": true
            }
        });

        let usage = parse_grok_billing(&payload).unwrap();

        assert_eq!(usage.windows.len(), 1);
        assert_eq!(usage.windows[0].id, "USAGE_PERIOD_TYPE_WEEKLY");
        assert_eq!(usage.windows[0].used_percent, Some(42.5));
        assert_eq!(usage.windows[0].window_seconds, Some(7 * 24 * 3600));
        assert_eq!(usage.windows[0].resets_at, Some(1780876800000));
        assert_eq!(usage.balances[0].amount, 12.5);
        assert_eq!(usage.balances[0].currency, UsageCurrency::Usd);
    }

    #[test]
    fn grok_zero_prepaid_balance_arrives_as_empty_object() {
        let payload = json!({ "config": { "creditUsagePercent": 0.0, "prepaidBalance": {} } });

        let usage = parse_grok_billing(&payload).unwrap();

        assert_eq!(usage.windows[0].id, "credits");
        assert_eq!(usage.windows[0].resets_at, None);
        assert_eq!(usage.balances[0].amount, 0.0);
    }

    #[test]
    fn grok_null_config_is_an_error() {
        assert!(parse_grok_billing(&json!({ "config": null })).is_err());
    }

    #[test]
    fn openrouter_key_with_limit() {
        let payload = json!({ "data": {
            "label": "sk-or-v1-xxx",
            "limit": 10.0,
            "limit_reset": "monthly",
            "limit_remaining": 7.5,
            "is_free_tier": false,
            "budgets": [{
                "scope": "key_limit", "guardrail_id": null, "budget_id": null,
                "limit_usd": 10.0, "spend_usd": 2.5, "remaining_usd": 7.5,
                "reset_interval": "monthly", "resets_at": "2026-12-01T00:00:00Z",
                "include_byok": false
            }],
            "usage": 2.5
        }});

        let usage = parse_openrouter_key(&payload).unwrap();

        assert_eq!(usage.plan.as_deref(), Some("paid"));
        assert_eq!(usage.windows[0].id, "keyLimit");
        assert_eq!(usage.windows[0].used_percent, Some(25.0));
        assert_eq!(usage.windows[0].used, Some(2.5));
        assert_eq!(usage.windows[0].limit, Some(10.0));
        assert_eq!(usage.windows[0].resets_at, Some(1796083200000));
        assert_eq!(usage.balances[0].amount, 7.5);
        assert_eq!(usage.balances[0].granted, Some(10.0));
    }

    #[test]
    fn openrouter_key_without_limit_has_only_the_plan() {
        let payload = json!({ "data": {
            "limit": null, "limit_remaining": null, "limit_reset": null,
            "is_free_tier": true, "budgets": [], "effective_budget": null
        }});

        let usage = parse_openrouter_key(&payload).unwrap();

        assert_eq!(usage.plan.as_deref(), Some("free"));
        assert!(usage.windows.is_empty());
        assert!(usage.balances.is_empty());
    }

    #[test]
    fn experiential_balance_is_total_minus_usage() {
        let payload = json!({ "data": { "total_credits": 12.5, "total_usage": 2.25 } });

        let usage = parse_experiential_credits(&payload).unwrap();

        assert_eq!(usage.balances[0].amount, 10.25);
        assert_eq!(usage.balances[0].granted, Some(12.5));
        assert_eq!(usage.balances[0].currency, UsageCurrency::Usd);
        assert!(usage.windows.is_empty());
    }

    #[test]
    fn experiential_answer_without_totals_is_an_error() {
        assert!(parse_experiential_credits(&json!({ "data": {} })).is_err());
    }
}
