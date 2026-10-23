use serde_json::{json, Value};

use crate::conversation::message::DocumentContent;

#[derive(Debug, Copy, Clone)]
pub enum DocumentFormat {
    OpenAi,
    Anthropic,
}

/// Media types that providers accept as native document input. Anything else is
/// reported back to the caller rather than being sent as an unusable blob.
pub const SUPPORTED_DOCUMENT_MEDIA_TYPES: [&str; 1] = ["application/pdf"];

pub fn document_media_type_is_supported(mime_type: &str) -> bool {
    SUPPORTED_DOCUMENT_MEDIA_TYPES.contains(&mime_type)
}

/// Video travels as a document whose media type is `video/*`. Only providers
/// that accept video input (OpenRouter, Gemini) send it; everyone else reports it
/// as unsupported like any other document type.
pub fn is_video_media_type(mime_type: &str) -> bool {
    mime_type.starts_with("video/")
}

/// OpenRouter's chat-completions video part (`video_url` with a data URL).
pub fn convert_video_url(document: &DocumentContent) -> Value {
    json!({
        "type": "video_url",
        "video_url": {
            "url": format!("data:{};base64,{}", document.mime_type, document.data),
        }
    })
}

pub fn convert_document(document: &DocumentContent, format: &DocumentFormat) -> Value {
    match format {
        DocumentFormat::OpenAi => json!({
            "type": "file",
            "file": {
                "filename": document.name.clone().unwrap_or_else(|| "document.pdf".to_string()),
                "file_data": format!("data:{};base64,{}", document.mime_type, document.data),
            }
        }),
        DocumentFormat::Anthropic => {
            let mut block = json!({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": document.mime_type,
                    "data": document.data,
                }
            });
            if let Some(name) = &document.name {
                block["title"] = json!(name);
            }
            block
        }
    }
}

/// Explains why a document was dropped so the model, and the caller reading the
/// request, can act on it instead of silently losing the attachment.
pub fn unsupported_document_text(document: &DocumentContent, reason: &str) -> String {
    match &document.name {
        Some(name) => format!(
            "[document \"{}\" ({}) not sent: {}]",
            name, document.mime_type, reason
        ),
        None => format!("[document ({}) not sent: {}]", document.mime_type, reason),
    }
}

pub const UNSUPPORTED_MEDIA_TYPE_REASON: &str =
    "only application/pdf documents can be sent to this provider";
pub const UNSUPPORTED_PROVIDER_REASON: &str = "this provider does not accept document input";
pub const ASSISTANT_ROLE_REASON: &str = "documents can only be sent in user messages";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_media_types_are_recognised() {
        assert!(is_video_media_type("video/mp4"));
        assert!(is_video_media_type("video/quicktime"));
        assert!(!is_video_media_type("application/pdf"));
        assert!(!is_video_media_type("image/png"));
    }

    #[test]
    fn convert_video_url_builds_a_data_url_part() {
        let video = DocumentContent::new("AAAA", "video/mp4").with_name("clip.mp4");
        assert_eq!(
            convert_video_url(&video),
            json!({
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,AAAA"}
            })
        );
    }

    #[test]
    fn unsupported_document_text_names_the_document_and_reason() {
        let named = DocumentContent::new("data", "text/csv").with_name("rows.csv");
        assert_eq!(
            unsupported_document_text(&named, UNSUPPORTED_MEDIA_TYPE_REASON),
            "[document \"rows.csv\" (text/csv) not sent: only application/pdf documents can be sent to this provider]"
        );

        let unnamed = DocumentContent::new("data", "text/csv");
        assert_eq!(
            unsupported_document_text(&unnamed, UNSUPPORTED_PROVIDER_REASON),
            "[document (text/csv) not sent: this provider does not accept document input]"
        );
    }
}
