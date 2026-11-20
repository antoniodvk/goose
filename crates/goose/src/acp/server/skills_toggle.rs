//! Toggle skills visibility (`SkillsState`) on an existing session without reloading it (A77).
//! Used by the Goose App when a chat enters or leaves "Nur Chat".

use super::*;

pub(super) const SET_SESSION_SKILLS_METHOD: &str = "_goose/unstable/session/skills/set";

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SetSessionSkillsParams {
    session_id: String,
    enabled: bool,
}

pub(super) fn is_set_session_skills_method(method: &str) -> bool {
    method == SET_SESSION_SKILLS_METHOD
}

impl GooseAcpAgent {
    pub(super) async fn on_set_session_skills(
        &self,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, agent_client_protocol::Error> {
        let req: SetSessionSkillsParams = serde_json::from_value(params).map_err(|error| {
            agent_client_protocol::Error::invalid_params().data(error.to_string())
        })?;
        let session_id = req.session_id.trim();
        if session_id.is_empty() {
            return Err(
                agent_client_protocol::Error::invalid_params().data("sessionId cannot be empty")
            );
        }

        let mut session = self
            .session_manager
            .get_session(session_id, false)
            .await
            .map_err(|_| {
                agent_client_protocol::Error::resource_not_found(Some(session_id.to_string()))
                    .data(format!("Session not found: {}", session_id))
            })?;

        set_skills_state(&mut session.extension_data, req.enabled)?;
        self.session_manager
            .update(&session.id)
            .extension_data(session.extension_data)
            .apply()
            .await
            .internal_err_ctx("Failed to update skills state")?;

        Ok(serde_json::json!({}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SkillsState;

    #[test]
    fn recognizes_set_session_skills_method() {
        assert!(is_set_session_skills_method(
            "_goose/unstable/session/skills/set"
        ));
        assert!(!is_set_session_skills_method(
            "_goose/unstable/session/skills/get"
        ));
    }

    #[test]
    fn validates_params_shape() {
        let ok: SetSessionSkillsParams = serde_json::from_value(serde_json::json!({
            "sessionId": "abc",
            "enabled": false
        }))
        .unwrap();
        assert_eq!(ok.session_id, "abc");
        assert!(!ok.enabled);

        let bad_bool = serde_json::from_value::<SetSessionSkillsParams>(serde_json::json!({
            "sessionId": "abc",
            "enabled": "no"
        }));
        assert!(bad_bool.is_err());

        let missing = serde_json::from_value::<SetSessionSkillsParams>(serde_json::json!({
            "sessionId": "abc"
        }));
        assert!(missing.is_err());
    }

    #[test]
    fn toggles_skills_state_roundtrip() {
        let mut data = ExtensionData::new();
        set_skills_state(&mut data, false).unwrap();
        assert_eq!(
            SkillsState::from_extension_data(&data),
            Some(SkillsState { enabled: false })
        );
        set_skills_state(&mut data, true).unwrap();
        assert_eq!(
            SkillsState::from_extension_data(&data),
            Some(SkillsState { enabled: true })
        );
    }
}
