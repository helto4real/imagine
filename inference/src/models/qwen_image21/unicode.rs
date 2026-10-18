//! Reference Python 3.12 Unicode 15 NFC and frozen regex character classes.
//! Tables are generated from the locked reference, not the host's Unicode data.
use serde_json::Value;
use std::{collections::BTreeMap, sync::OnceLock};

pub struct Unicode {
    decomposition: Vec<(u32, Vec<u32>)>,
    combining: Vec<(u32, u8)>,
    composition: BTreeMap<(u32, u32), u32>,
    properties: BTreeMap<String, Vec<(u32, u32)>>,
    contractions: BTreeMap<char, Vec<u32>>,
}
impl Unicode {
    pub fn frozen() -> &'static Self {
        static DATA: OnceLock<Unicode> = OnceLock::new();
        DATA.get_or_init(|| {
            let v: Value = serde_json::from_str(include_str!(
                "../../../resources/qwen-tokenizer/unicode.json"
            ))
            .expect("Generated frozen Unicode tables");
            let n = |v: &Value| v.as_u64().unwrap() as u32;
            Self {
                decomposition: v["decomposition"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| (n(&p[0]), p[1].as_array().unwrap().iter().map(n).collect()))
                    .collect(),
                combining: v["combining"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| (n(&p[0]), n(&p[1]) as u8))
                    .collect(),
                composition: v["composition"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| ((n(&p[0]), n(&p[1])), n(&p[2])))
                    .collect(),
                properties: v["properties"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .filter(|(k, _)| k.as_str() != "contractions")
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            v.as_array()
                                .unwrap()
                                .iter()
                                .map(|p| (n(&p[0]), n(&p[1])))
                                .collect(),
                        )
                    })
                    .collect(),
                contractions: v["properties"]["contractions"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.chars().next().unwrap(),
                            v.as_array().unwrap().iter().map(n).collect(),
                        )
                    })
                    .collect(),
            }
        })
    }
    fn cc(&self, c: u32) -> u8 {
        self.combining
            .binary_search_by_key(&c, |p| p.0)
            .map(|i| self.combining[i].1)
            .unwrap_or(0)
    }
    fn decompose(&self, c: u32, out: &mut Vec<u32>) {
        let s = c.wrapping_sub(0xAC00);
        if s < 11172 {
            out.push(0x1100 + s / 588);
            out.push(0x1161 + (s % 588) / 28);
            if !s.is_multiple_of(28) {
                out.push(0x11A7 + s % 28);
            }
        } else if let Ok(i) = self.decomposition.binary_search_by_key(&c, |p| p.0) {
            for &d in &self.decomposition[i].1 {
                self.decompose(d, out);
            }
        } else {
            out.push(c);
        }
    }
    fn compose(&self, a: u32, b: u32) -> Option<u32> {
        let l = a.wrapping_sub(0x1100);
        let v = b.wrapping_sub(0x1161);
        if l < 19 && v < 21 {
            return Some(0xAC00 + (l * 21 + v) * 28);
        }
        let s = a.wrapping_sub(0xAC00);
        let t = b.wrapping_sub(0x11A7);
        if s < 11172 && s.is_multiple_of(28) && t > 0 && t < 28 {
            return Some(a + t);
        }
        self.composition.get(&(a, b)).copied()
    }
    pub fn nfc(&self, text: &str) -> String {
        let mut decomposed = Vec::with_capacity(text.len());
        for c in text.chars() {
            self.decompose(c as u32, &mut decomposed);
        }
        // Canonical ordering is stable, and never crosses a class-zero starter.
        for i in 1..decomposed.len() {
            let cc = self.cc(decomposed[i]);
            if cc == 0 {
                continue;
            }
            let mut j = i;
            while j > 0 && self.cc(decomposed[j - 1]) > cc {
                decomposed.swap(j - 1, j);
                j -= 1;
            }
        }
        let mut out = Vec::with_capacity(decomposed.len());
        let mut starter = None;
        let mut last = 0;
        for c in decomposed {
            let cc = self.cc(c);
            if let Some(index) = starter
                && (last == 0 || last < cc)
                && let Some(composed) = self.compose(out[index], c)
            {
                out[index] = composed;
                continue;
            }
            if cc == 0 {
                starter = Some(out.len());
            }
            out.push(c);
            last = cc;
        }
        out.into_iter()
            .map(|c| char::from_u32(c).expect("Frozen scalar decomposition"))
            .collect()
    }
    pub fn property(&self, name: &str, c: char) -> bool {
        let ranges = &self.properties[name];
        let cp = c as u32;
        let i = ranges.partition_point(|p| p.0 <= cp);
        i > 0 && cp <= ranges[i - 1].1
    }
    fn contraction(&self, c: char, wanted: char) -> bool {
        self.contractions[&wanted].contains(&(c as u32))
    }
    /// The seven ordered alternatives of Qwen2's PRETOKENIZE_REGEX, using its
    /// exact frozen classes and greedy/backtracking boundaries.
    pub fn pretokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let chars = text.char_indices().collect::<Vec<_>>();
        let len = chars.len();
        let mut i = 0;
        let mut out = vec![];
        let l = |j: usize| self.property("letters", chars[j].1);
        let n = |j: usize| self.property("numbers", chars[j].1);
        let w = |j: usize| self.property("whitespace", chars[j].1);
        let newline = |j: usize| matches!(chars[j].1, '\r' | '\n');
        let bad = |j: usize| !w(j) && !l(j) && !n(j);
        while i < len {
            let start = i;
            let mut end = None;
            if chars[i].1 == '\'' {
                for contraction in ["s", "t", "re", "ve", "m", "ll", "d"] {
                    let expected = contraction.chars().collect::<Vec<_>>();
                    if i + expected.len() < len
                        && expected
                            .iter()
                            .enumerate()
                            .all(|(k, c)| self.contraction(chars[i + k + 1].1, *c))
                    {
                        end = Some(i + 1 + expected.len());
                        break;
                    }
                }
            }
            if end.is_none() {
                let mut j = i;
                if !newline(j) && !l(j) && !n(j) {
                    j += 1;
                }
                if j < len && l(j) {
                    while j < len && l(j) {
                        j += 1;
                    }
                    end = Some(j);
                }
            }
            if end.is_none() && n(i) {
                end = Some(i + 1);
            }
            if end.is_none() {
                let mut j = i + usize::from(chars[i].1 == ' ');
                if j < len && bad(j) {
                    while j < len && bad(j) {
                        j += 1;
                    }
                    while j < len && newline(j) {
                        j += 1;
                    }
                    end = Some(j);
                }
            }
            if end.is_none() && w(i) {
                let mut j = i;
                let mut last_newline = None;
                while j < len && w(j) {
                    if newline(j) {
                        last_newline = Some(j + 1);
                    }
                    j += 1;
                }
                end = last_newline.or_else(|| {
                    if j == len || j == i + 1 {
                        Some(j)
                    } else {
                        Some(j - 1)
                    }
                });
            }
            i = end.expect("The frozen regex covers every Unicode scalar");
            out.push(&text[chars[start].0..if i == len { text.len() } else { chars[i].0 }]);
        }
        out
    }
}
