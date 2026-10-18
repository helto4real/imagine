//! MOSS's original Qwen3.5 tokenizer wrapper over the shared Qwen2 BPE.
use crate::{
    models::qwen_image21::{QwenTokenizer, TokenizedPrompt, TokenizerOptions, Unicode},
    native::Error,
};

#[derive(Default)]
pub struct MossTokenizerOptions {
    pub common: TokenizerOptions,
    pub thinking: Option<bool>,
}
pub struct MossTokenizer {
    inner: QwenTokenizer,
}
impl MossTokenizer {
    pub fn frozen() -> Result<Self, Error> {
        Ok(Self {
            inner: QwenTokenizer::from_assets(
                include_str!("../../../resources/moss-tokenizer/vocab.json"),
                include_str!("../../../resources/moss-tokenizer/tokenizer_config.json"),
                include_str!("../../../resources/moss-tokenizer/merges.txt"),
                248044,
                248056,
            )?,
        })
    }
    pub fn encode(&mut self, text: &str) -> Result<Vec<i64>, Error> {
        self.inner.encode(text)
    }
    pub fn tokenize(
        &mut self,
        text: &str,
        options: &MossTokenizerOptions,
    ) -> Result<TokenizedPrompt, Error> {
        let opts = &options.common;
        let auto = opts.llama_template.is_none()
            && options.thinking.is_none()
            && !text.starts_with("<|im_start|>");
        let text = if auto {
            text.trim_matches(|c| Unicode::frozen().property("python_whitespace", c))
        } else {
            text
        };
        let skip = text.starts_with("<|im_start|>");
        let text = if opts.prevent_empty_text && text.is_empty() {
            " "
        } else {
            text
        };
        let custom = if auto {
            Some("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n")
        } else {
            opts.llama_template.as_deref()
        };
        let formatted = if skip {
            text.into()
        } else {
            let owned;
            let template = match custom {
                Some(x) => x,
                None => {
                    let vision = "<|vision_start|><|image_pad|><|vision_end|>".repeat(opts.images);
                    owned = format!(
                        "<|im_start|>user\n{vision}{{}}<|im_end|>\n<|im_start|>assistant\n"
                    );
                    &owned
                }
            };
            if template.matches("{}").count() != 1
                || template.replace("{}", "").contains(['{', '}'])
            {
                return Err(Error::BindingGap {
                    backend: "moss_tokenizer".into(),
                    operation: "General Python llama_template.format".into(),
                });
            }
            let mut formatted = template.replacen("{}", text, 1);
            if !opts.system_prompt.is_empty() {
                formatted = format!(
                    "<|im_start|>system\n{}<|im_end|>\n{}",
                    opts.system_prompt, formatted
                );
            }
            if custom.is_none() {
                formatted.push_str(if options.thinking == Some(true) {
                    "<think>\n"
                } else {
                    "<think>\n\n</think>\n\n"
                });
            }
            formatted
        };
        // The source SDTokenizer owns escaping, word groups and padding. The
        // shared implementation receives the already formatted raw chat turn.
        let raw = TokenizerOptions {
            skip_template: true,
            ..opts.clone()
        };
        self.inner.tokenize_prompt(&formatted, &raw)
    }
}
