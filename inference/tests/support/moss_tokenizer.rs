use imagine_inference::models::{
    moss_image::{MossTokenizer, MossTokenizerOptions},
    qwen_image21::{PromptToken, TokenizerOptions},
};
use serde_json::Value;
#[test]
fn moss_original_qwen35_tokenizer_bpe_trim_thinking_templates_images_and_padding_match() {
    let mut tokenizer = MossTokenizer::frozen().unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/moss-tokenizer.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 595);
    for (index, case) in cases.iter().enumerate() {
        let text = case["text"].as_str().unwrap();
        if case["operation"] == "raw" {
            assert_eq!(
                serde_json::json!(tokenizer.encode(text).unwrap()),
                case["ids"],
                "raw/{index}"
            );
            continue;
        }
        let o = &case["options"];
        let options = MossTokenizerOptions {
            thinking: o["thinking"].as_bool(),
            common: TokenizerOptions {
                images: o["images"].as_u64().unwrap_or(0) as usize,
                prevent_empty_text: o["prevent_empty_text"].as_bool().unwrap_or(false),
                system_prompt: o["system_prompt"].as_str().unwrap_or("").into(),
                llama_template: o["llama_template"].as_str().map(str::to_owned),
                min_length: o["min_length"].as_u64().unwrap_or(1) as usize,
                min_padding: o["min_padding"].as_u64().unwrap_or(0) as usize,
                max_length: o["max_length"].as_u64().unwrap_or(99_999_999) as usize,
                ..TokenizerOptions::default()
            },
        };
        let actual = tokenizer
            .tokenize(text, &options)
            .unwrap_or_else(|e| panic!("{index}: {e}"));
        let batches = case["batches"].as_array().unwrap();
        assert_eq!(actual.batches.len(), batches.len(), "{index}");
        for (batch, expected) in actual.batches.iter().zip(batches) {
            let expected = expected.as_array().unwrap();
            assert_eq!(batch.len(), expected.len(), "{index}");
            for (token, e) in batch.iter().zip(expected) {
                let id = if let Some(i) = e[0]["image"].as_u64() {
                    PromptToken::Image(i as usize)
                } else {
                    PromptToken::Id(e[0].as_i64().unwrap())
                };
                assert_eq!(token.token, id, "{index}");
                assert_eq!(token.weight.to_bits(), e[1].as_f64().unwrap().to_bits());
                if case["word_ids"] == true {
                    assert_eq!(token.word_id, e[2].as_u64().unwrap() as usize, "{index}");
                }
            }
        }
    }
}
