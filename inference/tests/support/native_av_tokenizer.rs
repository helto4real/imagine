use imagine_inference::{
    execution_value::Value as V,
    models::minimax_h3::tokenizer::{Reference, Token, Tokenizer},
};
use serde_json::{Value, json};
use tch::{Device, Kind, Tensor};
fn image(n: i64) -> V {
    V::tensor(
        Tensor::arange(n * 18, (Kind::Float, Device::Cpu))
            .reshape([n, 2, 3, 3])
            .g_div_scalar(127),
    )
}
#[test]
fn h3_raw_tokenizer_matches_original_labels_fixed_specials_reference_order_video_pairs_and_empty_fallback()
 {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/native-av-tokenizer.json")).unwrap();
    use sha2::{Digest, Sha256};
    for (name, bytes) in [
        (
            "vocab.json",
            include_bytes!("../../resources/qwen-tokenizer/vocab.json").as_slice(),
        ),
        (
            "merges.txt",
            include_bytes!("../../resources/qwen-tokenizer/merges.txt").as_slice(),
        ),
        (
            "tokenizer_config.json",
            include_bytes!("../../resources/qwen-tokenizer/tokenizer_config.json").as_slice(),
        ),
    ] {
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            f["assets"][name].as_str().unwrap()
        );
    }
    let mut tokenizer = Tokenizer::frozen().unwrap();
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 108);
    let first = image(1);
    let second = V::tensor(first.as_tensor().unwrap() + 1);
    let triple = image(3);
    let pair = image(2);
    let empty = image(0);
    for (index, case) in cases.iter().enumerate() {
        let images = match case["scenario"].as_str().unwrap() {
            "images" | "empty-refs-fallback" | "mixed" | "empty-video" => {
                vec![first.clone(), second.clone()]
            }
            _ => vec![],
        };
        let refs = match case["scenario"].as_str().unwrap() {
            "mixed" => vec![
                Reference::Audio,
                Reference::Image(&second),
                Reference::Video {
                    frames: &triple,
                    timestamps: None,
                },
                Reference::Image(&first),
                Reference::Audio,
            ],
            "timestamps" => vec![
                Reference::Video {
                    frames: &pair,
                    timestamps: Some(&[-0.25, 0.0]),
                },
                Reference::Video {
                    frames: &first,
                    timestamps: Some(&[2.25]),
                },
            ],
            "empty-video" => vec![Reference::Video {
                frames: &empty,
                timestamps: None,
            }],
            _ => vec![],
        };
        let entries = tokenizer
            .tokenize(
                case["text"].as_str().unwrap(),
                &images,
                &refs,
                case["word_ids"].as_bool().unwrap(),
            )
            .unwrap();
        let output=entries.iter().map(|e|{
   let token=match &e.token {
    Token::Id(i)=>json!(i),
    Token::Vision{data,video_block}=>{let t=data.as_tensor().unwrap();json!({"kind":"vision","shape":t.size(),"data":super::bits(&t).iter().map(|v|format!("{v:02x}")).collect::<String>(),"video_block":video_block,"original_type":"image"})}
   };
   let mut value=vec![token,json!(e.weight)];if let Some(word_id)=e.word_id{value.push(json!(word_id));}Value::Array(value)
  }).collect::<Vec<_>>();
        assert_eq!(
            json!(output),
            case["entries"],
            "H3 presentation case {index}: {}",
            case["scenario"]
        );
        for entry in entries {
            if let Token::Vision {
                data,
                video_block: false,
            } = entry.token
            {
                assert!(data.same_object(&first) || data.same_object(&second));
            }
        }
    }
}
