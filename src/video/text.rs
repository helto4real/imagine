//! Director text operations follow CPython's Unicode semantics.
use super::text_unicode::ALNUM;

pub(crate) fn word(c: char) -> bool {
    c == '_'
        || ALNUM
            .binary_search_by(|&(start, end)| {
                if (c as u32) < start {
                    std::cmp::Ordering::Greater
                } else if (c as u32) > end {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
}

pub(crate) use imagine_video_contract::references::python_space as space;

pub(crate) trait PythonText {
    fn pytrim(&self) -> &str;
}
impl PythonText for str {
    fn pytrim(&self) -> &str {
        self.trim_matches(space)
    }
}

pub(crate) fn single(s: &str) -> String {
    s.split(space)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn lines(s: &str) -> impl Iterator<Item = &str> {
    // Empty CRLF/trailing entries disappear in the caller, as in Director.
    s.split([
        '\n', '\r', '\u{b}', '\u{c}', '\u{1c}', '\u{1d}', '\u{1e}', '\u{85}', '\u{2028}',
        '\u{2029}',
    ])
}

pub(crate) fn ascii_case(c: char) -> char {
    match c {
        'İ' | 'ı' => 'i',
        'ſ' => 's',
        'K' => 'k',
        _ => c.to_ascii_lowercase(),
    }
}

/// Python re.IGNORECASE's extra matches for ASCII letter patterns.
pub(crate) fn prefix_case(text: &str, pattern: &str) -> Option<usize> {
    let mut chars = text.char_indices();
    let mut end = 0;
    for expected in pattern.chars() {
        let (index, actual) = chars.next()?;
        if ascii_case(actual) != expected {
            return None;
        }
        end = index + actual.len_utf8();
    }
    Some(end)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn director_unicode_boundaries_and_space() {
        assert!(word('é') && word('中') && word('²') && word('_'));
        assert!(!word('\u{345}') && !word('\u{5b0}') && !word('\u{1c}'));
        assert_eq!(single("\u{1c}Hej\u{1f}\u{a0}värld\u{3000}"), "Hej värld");
        assert_eq!("\u{1e}text\u{1f}".pytrim(), "text");
        assert_eq!(prefix_case("ſAYſ!", "says"), Some(6));
        assert_eq!(prefix_case("ın voiceover", "in"), Some(3));
    }
}
