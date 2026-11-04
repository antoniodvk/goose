//! Byte order marks of text files: detection, decoding and writing back in the same encoding.
//!
//! read shows UTF-16 files decoded; edit and write keep the encoding a file already has, so a
//! UTF-16 file (for example a Windows PowerShell script or a .reg file) stays UTF-16 after a change.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bom {
    /// No byte order mark: UTF-8 (or something else that is not text).
    None,
    Utf8,
    Utf16Le,
    Utf16Be,
}

impl Bom {
    /// The encoding announced by the first bytes, and the bytes after the mark.
    pub fn detect(bytes: &[u8]) -> (Bom, &[u8]) {
        if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
            (Bom::Utf8, rest)
        } else if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
            (Bom::Utf16Le, rest)
        } else if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
            (Bom::Utf16Be, rest)
        } else {
            (Bom::None, bytes)
        }
    }

    /// Longest byte order mark is three bytes.
    pub fn detect_in_file(path: &std::path::Path) -> Bom {
        use std::io::Read;
        let mut head = [0u8; 3];
        let read = std::fs::File::open(path)
            .and_then(|mut file| file.read(&mut head))
            .unwrap_or(0);
        Bom::detect(&head[..read]).0
    }

    /// Name for messages; `None` for plain UTF-8, which needs no mention.
    pub fn label(self) -> Option<&'static str> {
        match self {
            Bom::None => None,
            Bom::Utf8 => Some("UTF-8 with BOM"),
            Bom::Utf16Le => Some("UTF-16 LE"),
            Bom::Utf16Be => Some("UTF-16 BE"),
        }
    }

    pub fn is_utf16(self) -> bool {
        matches!(self, Bom::Utf16Le | Bom::Utf16Be)
    }

    /// Text as file bytes: byte order mark first, then the text in this encoding.
    pub fn encode(self, text: &str) -> Vec<u8> {
        match self {
            Bom::None => text.as_bytes().to_vec(),
            Bom::Utf8 => {
                let mut bytes = vec![0xEF, 0xBB, 0xBF];
                bytes.extend_from_slice(text.as_bytes());
                bytes
            }
            Bom::Utf16Le => {
                let mut bytes = vec![0xFF, 0xFE];
                bytes.extend(text.encode_utf16().flat_map(u16::to_le_bytes));
                bytes
            }
            Bom::Utf16Be => {
                let mut bytes = vec![0xFE, 0xFF];
                bytes.extend(text.encode_utf16().flat_map(u16::to_be_bytes));
                bytes
            }
        }
    }

    /// Text of the bytes after the mark, without replacing anything: a file that does not decode
    /// exactly must not be rewritten, because writing it back would change its content.
    pub fn decode_strict(self, body: &[u8]) -> Result<String, &'static str> {
        match self {
            Bom::None | Bom::Utf8 => std::str::from_utf8(body)
                .map(str::to_string)
                .map_err(|_| "the file is not valid UTF-8 (nor UTF-16 with a BOM)"),
            Bom::Utf16Le | Bom::Utf16Be => {
                let pairs = body.chunks_exact(2);
                if !pairs.remainder().is_empty() {
                    return Err("the UTF-16 file has an odd number of bytes");
                }
                let units: Vec<u16> = pairs
                    .map(|pair| {
                        let pair = [pair[0], pair[1]];
                        if self == Bom::Utf16Be {
                            u16::from_be_bytes(pair)
                        } else {
                            u16::from_le_bytes(pair)
                        }
                    })
                    .collect();
                String::from_utf16(&units).map_err(|_| "the UTF-16 file has an unpaired surrogate")
            }
        }
    }

    /// Text of the bytes after the mark for display; invalid parts become U+FFFD.
    pub fn decode_lossy(self, body: &[u8]) -> String {
        match self {
            Bom::None | Bom::Utf8 => String::from_utf8_lossy(body).into_owned(),
            Bom::Utf16Le | Bom::Utf16Be => {
                let units: Vec<u16> = body
                    .chunks_exact(2)
                    .map(|pair| {
                        let pair = [pair[0], pair[1]];
                        if self == Bom::Utf16Be {
                            u16::from_be_bytes(pair)
                        } else {
                            u16::from_le_bytes(pair)
                        }
                    })
                    .collect();
                String::from_utf16_lossy(&units)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_every_mark() {
        assert_eq!(Bom::detect(b"abc"), (Bom::None, &b"abc"[..]));
        assert_eq!(Bom::detect(&[0xEF, 0xBB, 0xBF, b'a']).0, Bom::Utf8);
        assert_eq!(Bom::detect(&[0xFF, 0xFE, b'a', 0]).0, Bom::Utf16Le);
        assert_eq!(Bom::detect(&[0xFE, 0xFF, 0, b'a']).0, Bom::Utf16Be);
        assert_eq!(Bom::detect(b"").0, Bom::None);
    }

    #[test]
    fn encode_then_decode_returns_the_text() {
        for bom in [Bom::None, Bom::Utf8, Bom::Utf16Le, Bom::Utf16Be] {
            let bytes = bom.encode("grün 😀\r\nzwei");
            let (found, body) = Bom::detect(&bytes);
            assert_eq!(found, bom);
            assert_eq!(found.decode_strict(body).unwrap(), "grün 😀\r\nzwei");
        }
    }

    #[test]
    fn strict_decoding_refuses_broken_files() {
        assert!(Bom::None.decode_strict(b"caf\xE9").is_err());
        assert!(Bom::Utf16Le.decode_strict(&[b'a', 0, b'b']).is_err());
        assert!(Bom::Utf16Le.decode_strict(&[0x00, 0xD8, b'a', 0]).is_err());
    }
}
