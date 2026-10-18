//! Slow Qwen2 byte BPE and Qwen Image 2.1 prompt templates, without Python.
use super::unicode::Unicode;
use crate::native::Error;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

pub const SYSTEM_PROMPT: &str =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
pub const VISION_BLOCK: &str = "<|vision_start|><|image_pad|><|vision_end|>";
pub struct QwenTokenizer {
    vocab: HashMap<String, i64>,
    merges: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    added: Vec<(String, i64)>,
    cache: HashMap<String, Vec<String>>,
}
fn gap(operation: &str) -> Error {
    Error::BindingGap {
        backend: "qwen_tokenizer".into(),
        operation: operation.into(),
    }
}
impl QwenTokenizer {
    pub fn frozen() -> Result<Self, Error> {
        let vocab: HashMap<String, i64> =
            serde_json::from_str(include_str!("../../../resources/qwen-tokenizer/vocab.json"))
                .map_err(|e| Error::Native(e.to_string()))?;
        let config: Value = serde_json::from_str(include_str!(
            "../../../resources/qwen-tokenizer/tokenizer_config.json"
        ))
        .map_err(|e| Error::Native(e.to_string()))?;
        let mut added: Vec<(String, i64)> = vec![];
        for (id, t) in config["added_tokens_decoder"]
            .as_object()
            .ok_or_else(|| gap("Missing frozen special tokens"))?
        {
            for name in ["lstrip", "rstrip", "single_word"] {
                if t[name].as_bool() == Some(true) {
                    return Err(gap("Extended AddedToken policies"));
                }
            }
            added.push((
                t["content"]
                    .as_str()
                    .ok_or_else(|| gap("Invalid special token"))?
                    .into(),
                id.parse().map_err(|_| gap("Invalid special token ID"))?,
            ));
        }
        // Trie selection chooses the earliest match and the longest token at
        // that position. Stable ordering also retains duplicate token identity.
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        let mut merges = HashMap::new();
        let mut rank = 0;
        for (i, line) in include_str!("../../../resources/qwen-tokenizer/merges.txt")
            .lines()
            .enumerate()
        {
            let line = line.trim();
            if line.is_empty() || (i == 0 && line.starts_with("#version:")) {
                continue;
            }
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() != 2 {
                return Err(gap("Non-pair BPE merge"));
            }
            merges.insert((parts[0].into(), parts[1].into()), rank);
            rank += 1;
        }
        let mut byte_encoder = ['\0'; 256];
        let mut extra = 256;
        for b in 0..256 {
            let natural =
                (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            byte_encoder[b as usize] = char::from_u32(if natural {
                b
            } else {
                let cp = extra;
                extra += 1;
                cp
            })
            .unwrap();
        }
        Ok(Self {
            vocab,
            merges,
            byte_encoder,
            added,
            cache: HashMap::new(),
        })
    }
    fn bpe(&mut self, token: String) -> Vec<String> {
        if let Some(pieces) = self.cache.get(&token) {
            return pieces.clone();
        }
        let mut word = token.chars().map(|c| c.to_string()).collect::<Vec<_>>();
        // The original singleton path returns before populating its cache.
        if word.len() < 2 {
            return word;
        }
        loop {
            let best = word
                .windows(2)
                .filter_map(|p| {
                    self.merges
                        .get(&(p[0].clone(), p[1].clone()))
                        .map(|rank| (*rank, (p[0].clone(), p[1].clone())))
                })
                .min_by_key(|p| p.0);
            let Some((_, pair)) = best else {
                break;
            };
            let mut next = vec![];
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
                    next.push(format!("{}{}", word[i], word[i + 1]));
                    i += 2;
                } else {
                    next.push(word[i].clone());
                    i += 1;
                }
            }
            word = next;
            if word.len() == 1 {
                break;
            }
        }
        self.cache.insert(token, word.clone());
        word
    }
    fn ordinary(&mut self, text: &str, out: &mut Vec<i64>) -> Result<(), Error> {
        for segment in Unicode::frozen().pretokenize(text) {
            let mapped = segment
                .as_bytes()
                .iter()
                .map(|b| self.byte_encoder[*b as usize])
                .collect();
            for piece in self.bpe(mapped) {
                let id = self
                    .vocab
                    .get(&piece)
                    .or_else(|| self.vocab.get("<|endoftext|>"))
                    .ok_or_else(|| gap("Missing byte-BPE vocabulary entry"))?;
                out.push(*id);
            }
        }
        Ok(())
    }
    pub fn encode(&mut self, text: &str) -> Result<Vec<i64>, Error> {
        let text = Unicode::frozen().nfc(text);
        let mut out = vec![];
        let mut ordinary = 0;
        let mut i = 0;
        while i < text.len() {
            let special = self
                .added
                .iter()
                .find(|(token, _)| text[i..].starts_with(token))
                .map(|(token, id)| (token.len(), *id));
            if let Some((length, id)) = special {
                if ordinary < i {
                    self.ordinary(&text[ordinary..i], &mut out)?;
                }
                out.push(id);
                i += length;
                ordinary = i;
            } else {
                i += text[i..].chars().next().unwrap().len_utf8();
            }
        }
        if ordinary < text.len() {
            self.ordinary(&text[ordinary..], &mut out)?;
        }
        Ok(out)
    }
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PromptToken {
    Id(i64),
    Image(usize),
}
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedToken {
    pub token: PromptToken,
    pub weight: f64,
    pub word_id: usize,
}
pub struct TokenizedPrompt {
    pub batches: Vec<Vec<WeightedToken>>,
    pub keep_vision: bool,
}
pub struct TokenizerOptions {
    pub images: usize,
    pub prevent_empty_text: bool,
    pub thinking: bool,
    pub keep_vision: bool,
    pub skip_template: bool,
    pub system_prompt: String,
    pub llama_template: Option<String>,
    pub min_length: usize,
    pub min_padding: usize,
    pub max_length: usize,
}
impl Default for TokenizerOptions {
    fn default() -> Self {
        Self {
            images: 0,
            prevent_empty_text: false,
            thinking: true,
            keep_vision: false,
            skip_template: false,
            system_prompt: String::new(),
            llama_template: None,
            min_length: 1,
            min_padding: 0,
            max_length: 99_999_999,
        }
    }
}
fn template(text: &str, options: &TokenizerOptions) -> Result<String, Error> {
    let skip = options.skip_template || text.starts_with("<|im_start|>");
    let text = if options.prevent_empty_text && text.is_empty() {
        " "
    } else {
        text
    };
    if skip {
        return Ok(text.into());
    }
    let mut result = if let Some(template) = &options.llama_template {
        // General Python format specifications have their own pending binding;
        // never interpret a different templating language as a fallback.
        if template.matches("{}").count() != 1 || template.replace("{}", "").contains(['{', '}']) {
            return Err(gap("General Python llama_template.format"));
        }
        template.replacen("{}", text, 1)
    } else {
        let references = (0..options.images)
            .map(|i| format!("<image{}>{VISION_BLOCK}", i + 1))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "{SYSTEM_PROMPT}<|im_start|>user\n{references}{text}<|im_end|>\n<|im_start|>assistant\n"
        )
    };
    if !options.system_prompt.is_empty() {
        let start = result
            .find("<|im_start|>user")
            .ok_or_else(|| Error::ReferenceException {
                class: "ValueError",
                message: "substring not found".into(),
            })?;
        result = format!(
            "<|im_start|>system\n{}<|im_end|>\n{}",
            options.system_prompt,
            &result[start..]
        );
    }
    if !options.thinking {
        result.push_str("<think>\n\n</think>\n\n");
    }
    Ok(result)
}
impl QwenTokenizer {
    /// QwenImage21Tokenizer -> Qwen3VLTokenizer -> SDTokenizer with weights
    /// disabled. Image entries are opaque caller-owned references, never files.
    pub fn tokenize_prompt(
        &mut self,
        text: &str,
        options: &TokenizerOptions,
    ) -> Result<TokenizedPrompt, Error> {
        let text = template(text, options)?
            .replace("\\)", "\0\u{1}")
            .replace("\\(", "\0\u{2}")
            .replace("\0\u{1}", ")")
            .replace("\0\u{2}", "(");
        let mut starts = vec![0];
        for (i, c) in text.char_indices() {
            let end = i + c.len_utf8();
            if Unicode::frozen().property("python_whitespace", c)
                && text[end..].starts_with("embedding:")
            {
                starts.push(end);
            }
        }
        starts.push(text.len());
        let mut groups = vec![];
        for range in starts.windows(2) {
            if range[0] != range[1] {
                groups.push(self.encode(&text[range[0]..range[1]])?);
            }
        }
        if options.max_length == 0 {
            return Err(gap("Zero-length SDTokenizer batching"));
        }
        let mut batches = vec![vec![]];
        for (word, group) in groups.into_iter().enumerate() {
            let mut offset = 0;
            let large = group.len() >= 8;
            while offset < group.len() {
                let batch = batches.last_mut().unwrap();
                let remaining = options.max_length.saturating_sub(batch.len());
                let count = if group.len() - offset > remaining {
                    if large { remaining } else { 0 }
                } else {
                    group.len() - offset
                };
                batch.extend(
                    group[offset..offset + count]
                        .iter()
                        .map(|id| WeightedToken {
                            token: PromptToken::Id(*id),
                            weight: 1.,
                            word_id: word + 1,
                        }),
                );
                offset += count;
                if offset < group.len() {
                    batches.push(vec![]);
                }
            }
        }
        let last = batches.last_mut().unwrap();
        // SDTokenizer's explicit minimum padding appends even when the existing
        // sequence already exceeds min_length.
        last.extend((0..options.min_padding).map(|_| WeightedToken {
            token: PromptToken::Id(151643),
            weight: 1.,
            word_id: 0,
        }));
        if last.len() < options.min_length {
            last.extend((last.len()..options.min_length).map(|_| WeightedToken {
                token: PromptToken::Id(151643),
                weight: 1.,
                word_id: 0,
            }));
        }
        let mut image = 0;
        for token in batches.iter_mut().flatten() {
            if token.token == PromptToken::Id(151655) && image < options.images {
                token.token = PromptToken::Image(image);
                image += 1;
            }
        }
        Ok(TokenizedPrompt {
            batches,
            keep_vision: options.keep_vision,
        })
    }
    pub fn vocabulary(&self) -> BTreeMap<&str, i64> {
        self.vocab
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .chain(self.added.iter().map(|(k, v)| (k.as_str(), *v)))
            .collect()
    }
}
