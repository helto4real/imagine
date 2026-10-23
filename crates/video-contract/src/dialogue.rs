//! Director's dialogue warnings for authored text: an incomplete `@name verb`
//! macro, quoted words after `@name` without a delivery word, and quoted text
//! with no speaker. They are advisory; compilation still treats the text as written.

/// A dialogue warning; `token` is the matched text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DialogueIssue {
    pub code: &'static str,
    pub token: String,
}
impl DialogueIssue {
    pub fn message(&self) -> &'static str {
        match self.code {
            INVALID_SYNTAX => {
                "Dialogue must use @name plus a supported delivery word (say, speak, ask, \
                 reply, exclaim, whisper, murmur, mutter, shout, yell, scream, or sing), an \
                 optional off-screen or in voiceover qualifier, an optional [Language], and \
                 double-quoted words."
            }
            SPEECH_VERB_REQUIRED => {
                "Quoted dialogue after @name requires a supported delivery word, for example \
                 @tomas whisper \"Hello\"."
            }
            _ => {
                "Quoted text may be unintended dialogue because it has no @name plus a \
                 supported delivery word, for example @tomas says \"Hello\"."
            }
        }
    }
}
pub const INVALID_SYNTAX: &str = "dialogue_macro_invalid_syntax";
pub const SPEECH_VERB_REQUIRED: &str = "dialogue_speech_verb_required";
pub const QUOTED_TEXT_WITHOUT_SPEAKER: &str = "dialogue_quoted_text_without_speaker";

const VERBS: [&str; 36] = [
    "say",
    "says",
    "saying",
    "speak",
    "speaks",
    "speaking",
    "ask",
    "asks",
    "asking",
    "reply",
    "replies",
    "replying",
    "exclaim",
    "exclaims",
    "exclaiming",
    "whisper",
    "whispers",
    "whispering",
    "murmur",
    "murmurs",
    "murmuring",
    "mutter",
    "mutters",
    "muttering",
    "shout",
    "shouts",
    "shouting",
    "yell",
    "yells",
    "yelling",
    "scream",
    "screams",
    "screaming",
    "sing",
    "sings",
    "singing",
];

/// A cursor over `text` by byte offset, mirroring Director's regular expressions.
struct Scan<'a> {
    text: &'a str,
}
impl Scan<'_> {
    fn next(&self, at: usize) -> Option<char> {
        self.text[at..].chars().next()
    }
    /// `\s+`
    fn space(&self, at: usize) -> Option<usize> {
        let mut p = at;
        while let Some(c) = self.next(p).filter(|c| c.is_whitespace()) {
            p += c.len_utf8();
        }
        (p > at).then_some(p)
    }
    fn word(c: char) -> bool {
        c.is_alphanumeric() || c == '_'
    }
    /// A case-insensitive ASCII literal.
    fn literal(&self, at: usize, word: &str) -> Option<usize> {
        let end = at + word.len();
        (self.text.len() >= end
            && self.text.is_char_boundary(end)
            && self.text[at..end].eq_ignore_ascii_case(word))
        .then_some(end)
    }
    /// `(?<![A-Za-z0-9_@])@[A-Za-z][A-Za-z0-9_-]*`
    fn alias(&self, at: usize) -> Option<usize> {
        if self.next(at) != Some('@')
            || self.text[..at]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '@'))
        {
            return None;
        }
        let mut p = at + 1;
        self.next(p).filter(char::is_ascii_alphabetic)?;
        p += 1;
        while self
            .next(p)
            .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
        {
            p += 1;
        }
        Some(p)
    }
    /// A delivery verb followed by a word boundary, longest first.
    fn verb(&self, at: usize) -> Option<usize> {
        let mut verbs = VERBS;
        verbs.sort_by_key(|v| std::cmp::Reverse(v.len()));
        verbs.iter().find_map(|verb| {
            let end = self.literal(at, verb)?;
            (!self.next(end).is_some_and(Self::word)).then_some(end)
        })
    }
    /// `\s+(off-screen|in\s+voiceover)`
    fn context(&self, at: usize) -> Option<usize> {
        let p = self.space(at)?;
        self.literal(p, "off-screen").or_else(|| {
            let p = self.literal(p, "in")?;
            self.literal(self.space(p)?, "voiceover")
        })
    }
    /// `\s+\[[^\]\r\n]+\]`
    fn language(&self, at: usize) -> Option<usize> {
        let p = self.space(at)?;
        (self.next(p) == Some('[')).then_some(())?;
        let mut q = p + 1;
        while let Some(c) = self.next(q).filter(|c| !matches!(c, ']' | '\r' | '\n')) {
            q += c.len_utf8();
        }
        (q > p + 1 && self.next(q) == Some(']')).then_some(q + 1)
    }
    /// `"(?:[^"\\\r\n]|\\.)*"`, returning its end and unescaped content.
    fn quote(&self, at: usize) -> Option<(usize, String)> {
        (self.next(at) == Some('"')).then_some(())?;
        let mut p = at + 1;
        let mut content = String::new();
        loop {
            let c = self.next(p)?;
            match c {
                '"' => return Some((p + 1, content)),
                '\r' | '\n' => return None,
                '\\' => {
                    let escaped = self.next(p + 1)?;
                    content.push(escaped);
                    p += 1 + escaped.len_utf8();
                }
                _ => {
                    content.push(c);
                    p += c.len_utf8();
                }
            }
        }
    }
    /// `\s+"…"` after `at`.
    fn spaced_quote(&self, at: usize) -> Option<usize> {
        Some(self.quote(self.space(at)?)?.0)
    }
    /// A complete macro: alias, verb, optional context and language, quote.
    fn macro_end(&self, at: usize) -> Option<usize> {
        let verb = self.verb(self.space(self.alias(at)?)?)?;
        let contexts = [self.context(verb), None];
        contexts.into_iter().flatten().chain([verb]).find_map(|p| {
            self.language(p)
                .and_then(|q| self.spaced_quote(q))
                .or_else(|| self.spaced_quote(p))
        })
    }
}

/// Dialogue warnings in `text`, in text order.
pub fn issues(text: &str) -> Vec<DialogueIssue> {
    let scan = Scan { text };
    let starts: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let mut occupied: Vec<(usize, usize)> = vec![];
    let mut p = 0;
    while p < text.len() {
        if let Some(end) = scan.macro_end(p) {
            occupied.push((p, end));
            p = end;
        } else {
            p += scan.next(p).map_or(1, char::len_utf8);
        }
    }
    let inside =
        |at: usize, spans: &[(usize, usize)]| spans.iter().any(|(s, e)| *s <= at && at < *e);
    let mut found: Vec<(usize, DialogueIssue)> = vec![];
    for &at in &starts {
        if let Some(end) = scan.alias(at).and_then(|p| scan.verb(scan.space(p)?))
            && !inside(at, &occupied)
        {
            found.push((
                at,
                DialogueIssue {
                    code: INVALID_SYNTAX,
                    token: text[at..end].into(),
                },
            ));
        }
    }
    for &at in &starts {
        if inside(at, &occupied) {
            continue;
        }
        let end = scan.alias(at).and_then(|p| {
            scan.language(p)
                .and_then(|q| scan.spaced_quote(q))
                .or_else(|| scan.spaced_quote(p))
        });
        if let Some(end) = end {
            occupied.push((at, end));
            found.push((
                at,
                DialogueIssue {
                    code: SPEECH_VERB_REQUIRED,
                    token: text[at..end].into(),
                },
            ));
        }
    }
    let mut p = 0;
    while p < text.len() {
        match scan.quote(p) {
            Some((end, content)) if !inside(p, &occupied) => {
                if !content.trim().is_empty() {
                    found.push((
                        p,
                        DialogueIssue {
                            code: QUOTED_TEXT_WITHOUT_SPEAKER,
                            token: text[p..end].into(),
                        },
                    ));
                }
                p = end;
            }
            Some((end, _)) => p = end,
            None => p += scan.next(p).map_or(1, char::len_utf8),
        }
    }
    found.sort_by_key(|(at, issue)| (*at, issue.code));
    found.into_iter().map(|(_, issue)| issue).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn codes(text: &str) -> Vec<&'static str> {
        issues(text).into_iter().map(|i| i.code).collect()
    }
    #[test]
    fn complete_macros_are_silent() {
        for text in [
            "@tomas says \"Hello\"",
            "@tomas whispers off-screen [English] \"Hi \\\"there\\\"\"",
            "@Ann  SHOUTS in   voiceover \"Run!\" and walks.",
            "@ann asks [Svenska] \"Hej?\"",
            "No dialogue here, mail tomas@example.com.",
            "Empty quote \"   \" is ignored.",
        ] {
            assert!(codes(text).is_empty(), "{text}: {:?}", issues(text));
        }
    }
    #[test]
    fn director_warnings_are_reported_in_text_order() {
        assert_eq!(codes("@tomas says Hello"), [INVALID_SYNTAX]);
        assert_eq!(codes("@tomas \"Hello\""), [SPEECH_VERB_REQUIRED]);
        assert_eq!(codes("@tomas [English] \"Hello\""), [SPEECH_VERB_REQUIRED]);
        assert_eq!(codes("She hears \"Hello\"."), [QUOTED_TEXT_WITHOUT_SPEAKER]);
        assert_eq!(
            codes("\"Hi\" then @a says \"Ok\" then @b sings"),
            [QUOTED_TEXT_WITHOUT_SPEAKER, INVALID_SYNTAX]
        );
        assert_eq!(issues("@tomas says Hello")[0].token, "@tomas says");
    }
}
