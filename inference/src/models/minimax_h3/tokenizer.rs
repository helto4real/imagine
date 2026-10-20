//! H3 raw text/vision token presentation. Local Qwen2 BPE, no chat template.
//! Typed reference requests only; textual-inversion loading is a separate binding.
use crate::{execution_value::Value, models::qwen_image21::QwenTokenizer, native::Error};

pub const EXTRA_TOKENS: [(&str, i64); 7] = [
    ("<d>", 151669),
    ("</d>", 151670),
    ("<|cutoff|>", 151671),
    ("<|lyrics_start|>", 151672),
    ("<|lyrics_end|>", 151673),
    ("<|caption_start|>", 151674),
    ("<|caption_end|>", 151675),
];
#[derive(Clone)]
pub enum Token {
    Id(i64),
    Vision { data: Value, video_block: bool },
}
#[derive(Clone)]
pub struct Entry {
    pub token: Token,
    pub weight: f64,
    pub word_id: Option<i64>,
}
pub enum Reference<'a> {
    Image(&'a Value),
    Audio,
    Video {
        frames: &'a Value,
        timestamps: Option<&'a [f64]>,
    },
}
pub struct Tokenizer {
    bpe: QwenTokenizer,
}
impl Tokenizer {
    pub fn frozen() -> Result<Self, Error> {
        // The locked H3 Qwen25 assets are byte-identical to this existing asset set.
        // Only this factory augments the fixed special IDs; Qwen Image stays unchanged.
        let mut config: serde_json::Value = serde_json::from_str(include_str!(
            "../../../resources/qwen-tokenizer/tokenizer_config.json"
        ))
        .map_err(|e| Error::Native(e.to_string()))?;
        for (name, id) in EXTRA_TOKENS {
            config["added_tokens_decoder"][id.to_string()] = serde_json::json!({"content":name,"lstrip":false,"rstrip":false,"single_word":false,"normalized":false,"special":true});
        }
        Ok(Self {
            bpe: QwenTokenizer::from_assets(
                include_str!("../../../resources/qwen-tokenizer/vocab.json"),
                &config.to_string(),
                include_str!("../../../resources/qwen-tokenizer/merges.txt"),
                151643,
                151655,
            )?,
        })
    }
    fn text(&mut self, text: &str, out: &mut Vec<Entry>) -> Result<(), Error> {
        if text.is_empty() {
            return Ok(());
        }
        // Original SDTokenizer disable_weights still escapes/unescapes parentheses,
        // and splits before whitespace-followed embedding: markers without a directory.
        let text = text.replace("\\)", "\0\u{1}").replace("\\(", "\0\u{2}");
        let text = text.replace("\0\u{1}", ")").replace("\0\u{2}", "(");
        let begin = out.len();
        let mut start = 0;
        for (index, _) in text.match_indices("embedding:") {
            if index > 0
                && text[..index]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
            {
                self.segment(&text[start..index], out)?;
                start = index;
            }
        }
        self.segment(&text[start..], out)?;
        if out.len() - begin > 99_999_999 {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "MiniMax H3 text segment exceeds the supported prompt length.".into(),
            });
        }
        Ok(())
    }
    fn segment(&mut self, text: &str, out: &mut Vec<Entry>) -> Result<(), Error> {
        if text.is_empty() {
            return Ok(());
        }
        let ids = self.bpe.encode(text)?;
        out.extend(ids.into_iter().map(|id| Entry {
            token: Token::Id(id),
            weight: 1.0,
            word_id: None,
        }));
        Ok(())
    }
    fn vision(data: Value, video_block: bool, out: &mut Vec<Entry>) {
        for token in [
            Token::Id(151652),
            Token::Vision { data, video_block },
            Token::Id(151653),
        ] {
            out.push(Entry {
                token,
                weight: 1.0,
                word_id: None,
            });
        }
    }
    /// Ref items take precedence only when nonempty, just as in the source.
    /// Audio contributes a text label, never a Qwen vision/audio embedding.
    pub fn tokenize(
        &mut self,
        text: &str,
        images: &[Value],
        refs: &[Reference<'_>],
        word_ids: bool,
    ) -> Result<Vec<Entry>, Error> {
        let mut out = vec![];
        if refs.is_empty() {
            for (i, image) in images.iter().enumerate() {
                self.text(&format!("<Picture {}>: ", i + 1), &mut out)?;
                Self::vision(image.clone(), false, &mut out);
            }
        } else {
            let (mut image, mut audio, mut video) = (0, 0, 0);
            for reference in refs {
                match reference {
                    Reference::Image(data) => {
                        image += 1;
                        self.text(&format!("<Picture {image}>: "), &mut out)?;
                        Self::vision((*data).clone(), false, &mut out);
                    }
                    Reference::Audio => {
                        audio += 1;
                        self.text(&format!("<Audio {audio}>: "), &mut out)?;
                    }
                    Reference::Video { frames, timestamps } => {
                        video += 1;
                        let mut frames = frames.as_tensor()?;
                        let n = *frames.size().first().ok_or_else(|| Error::BindingGap {
                            backend: "minimax_h3".into(),
                            operation: "video reference requires temporal dimension".into(),
                        })?;
                        let mut times = timestamps
                            .map(<[f64]>::to_vec)
                            .unwrap_or_else(|| (0..n).map(|i| i as f64 / 2.0).collect());
                        if n % 2 == 1 {
                            frames = tch::Tensor::f_cat(
                                &[&frames, &frames.f_slice(0, n - 1, n, 1)?],
                                0,
                            )?;
                            times.push(*times.last().ok_or_else(index_error)?);
                        }
                        self.text(&format!("<Video {video}>: "), &mut out)?;
                        for i in (0..frames.size()[0]).step_by(2) {
                            let ts = (times.get(i as usize).ok_or_else(index_error)?
                                + times.get(i as usize + 1).ok_or_else(index_error)?)
                                / 2.0;
                            let label = if ts.is_nan() {
                                "nan".to_string()
                            } else {
                                format!("{ts:.1}")
                            };
                            self.text(&format!("<{label} seconds>"), &mut out)?;
                            Self::vision(
                                Value::tensor(frames.f_slice(0, i, i + 2, 1)?),
                                true,
                                &mut out,
                            );
                        }
                    }
                }
            }
        }
        self.text(text, &mut out)?;
        if out.is_empty() {
            out.push(Entry {
                token: Token::Id(151643),
                weight: 1.0,
                word_id: None,
            });
        }
        if word_ids {
            for entry in &mut out {
                entry.word_id = Some(0);
            }
        }
        Ok(out)
    }
}
fn index_error() -> Error {
    Error::ReferenceException {
        class: "IndexError",
        message: "list index out of range".into(),
    }
}
