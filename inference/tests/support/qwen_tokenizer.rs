use imagine_inference::models::qwen_image21::{
    PromptToken, QwenTokenizer, TokenizerOptions, Unicode,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
fn fixture() -> Value {
    serde_json::from_str(include_str!("../../fixtures/qwen-image21-tokenizer.json")).unwrap()
}
#[test]
fn qwen_image21_original_tokenizer_ids_templates_padding_images_and_unicode_match() {
    let mut tokenizer = QwenTokenizer::frozen().unwrap();
    let data = fixture();
    for (index, c) in data["cases"].as_array().unwrap().iter().enumerate() {
        let text = c["text"].as_str().unwrap();
        if c["operation"] == "raw" {
            assert_eq!(
                Unicode::frozen().nfc(text),
                c["normalized"].as_str().unwrap(),
                "NFC/{index}"
            );
            assert_eq!(
                Unicode::frozen().pretokenize(text),
                c["segments"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect::<Vec<_>>(),
                "regex/{index}"
            );
            let ids = tokenizer.encode(text).unwrap();
            assert_eq!(
                ids,
                c["ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect::<Vec<_>>(),
                "BPE/{index}"
            );
            continue;
        }
        let o = &c["options"];
        let mut options = TokenizerOptions::default();
        if let Some(n) = o["images"].as_u64() {
            options.images = n as usize;
        }
        if let Some(n) = o["min_length"].as_u64() {
            options.min_length = n as usize;
        }
        if let Some(n) = o["min_padding"].as_u64() {
            options.min_padding = n as usize;
        }
        if let Some(n) = o["max_length"].as_u64() {
            options.max_length = n as usize;
        }
        if let Some(v) = o["thinking"].as_bool() {
            options.thinking = v;
        }
        if let Some(v) = o["keep_vision"].as_bool() {
            options.keep_vision = v;
        }
        if let Some(v) = o["prevent_empty_text"].as_bool() {
            options.prevent_empty_text = v;
        }
        if let Some(v) = o["skip_template"].as_bool() {
            options.skip_template = v;
        }
        if let Some(v) = o["system_prompt"].as_str() {
            options.system_prompt = v.into();
        }
        options.llama_template = o["llama_template"].as_str().map(str::to_owned);
        let result = tokenizer.tokenize_prompt(text, &options);
        if let Some(expected) = c["error"].as_str() {
            match result
                .err()
                .unwrap_or_else(|| panic!("{index}: expected {expected}"))
            {
                imagine_inference::native::Error::ReferenceException { class, message } => {
                    assert_eq!(class, expected);
                    assert_eq!(message, c["message"].as_str().unwrap());
                }
                other => panic!("{index}: {other:?}"),
            }
            continue;
        }
        let actual = result.unwrap_or_else(|e| panic!("prompt/{index}: {e}"));
        assert_eq!(actual.keep_vision, c["keep_vision"].as_bool().unwrap());
        assert_eq!(
            actual.batches.len(),
            c["batches"].as_array().unwrap().len(),
            "prompt/{index}"
        );
        for (batch, expected) in actual.batches.iter().zip(c["batches"].as_array().unwrap()) {
            let expected = expected.as_array().unwrap();
            assert_eq!(batch.len(), expected.len(), "prompt/{index}");
            for (a, e) in batch.iter().zip(expected) {
                let token = if let Some(image) = e[0]["image"].as_u64() {
                    PromptToken::Image(image as usize)
                } else {
                    PromptToken::Id(e[0].as_i64().unwrap())
                };
                assert_eq!(a.token, token, "prompt/{index}");
                assert_eq!(a.weight.to_bits(), e[1].as_f64().unwrap().to_bits());
                if c["word_ids"].as_bool().unwrap() {
                    assert_eq!(a.word_id, e[2].as_u64().unwrap() as usize, "word/{index}");
                }
            }
        }
    }
}
#[test]
fn qwen_image21_original_all_unicode_scalar_nfc_and_contextual_regex_hashes_match() {
    let tables: Value =
        serde_json::from_str(include_str!("../../resources/qwen-tokenizer/unicode.json")).unwrap();
    let fixture = fixture();
    let mut nfc = Sha256::new();
    let mut regex = Sha256::new();
    let mut count = 0_u64;
    for cp in 0..=0x10FFFF {
        let Some(c) = char::from_u32(cp) else {
            continue;
        };
        let normalized = Unicode::frozen().nfc(&c.to_string());
        nfc.update((normalized.len() as u32).to_le_bytes());
        nfc.update(normalized.as_bytes());
        let text = format!("a{c}9 {c}x\r\n'{c}s");
        let pieces = Unicode::frozen().pretokenize(&text);
        regex.update((pieces.len() as u32).to_le_bytes());
        for piece in pieces {
            regex.update((piece.len() as u32).to_le_bytes());
            regex.update(piece.as_bytes());
        }
        count += 1;
    }
    assert_eq!(count, tables["scalar_count"].as_u64().unwrap());
    assert_eq!(count, fixture["scalar_count"].as_u64().unwrap());
    assert_eq!(
        format!("{:x}", nfc.finalize()),
        tables["all_scalar_nfc_sha256"].as_str().unwrap()
    );
    assert_eq!(
        format!("{:x}", regex.finalize()),
        fixture["all_scalar_regex_sha256"].as_str().unwrap()
    );
}
