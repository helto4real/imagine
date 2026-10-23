//! Director timeline reference token semantics, shared by source capture and fingerprints.
use std::collections::BTreeSet;
pub fn python_space(c: char) -> bool {
    matches!(c, '\u{9}'..='\u{d}' | '\u{1c}'..='\u{20}' | '\u{85}' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}
pub struct Token {
    pub start: usize,
    pub end: usize,
    pub alias: String,
}
pub fn aliases(text: &str) -> BTreeSet<String> {
    tokens(text).into_iter().map(|t| t.alias).collect()
}
pub fn tokens(text: &str) -> Vec<Token> {
    let bytes = text.as_bytes();
    let mut found = vec![];
    let mut pos = 0;
    while pos < bytes.len() {
        if bytes[pos] != b'@'
            || pos > 0
                && (bytes[pos - 1].is_ascii_alphanumeric() || b"_@".contains(&bytes[pos - 1]))
            || !bytes.get(pos + 1).is_some_and(u8::is_ascii_alphabetic)
        {
            pos += 1;
            continue;
        }
        let token_start = pos;
        let start = pos + 1;
        pos = start + 1;
        while bytes
            .get(pos)
            .is_some_and(|b| b.is_ascii_alphanumeric() || b"_-".contains(b))
        {
            pos += 1;
        }
        let alias = text[start..pos].to_ascii_lowercase();
        // A complete optional strength belongs to this token, including any @
        // within it. An unclosed bracket does not consume the following text.
        if bytes.get(pos) == Some(&b'[')
            && let Some(end) = bytes[pos + 1..].iter().position(|b| *b == b']')
        {
            pos += end + 2;
        }
        found.push(Token {
            start: token_start,
            end: pos,
            alias,
        });
    }
    found
}
