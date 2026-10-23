use imagine_server::{contract::JobRequest, h3_protocol as h3, video};
use serde_json::json;
#[test]
fn profile_contract_requires_exactly_one_start_image_and_no_other_references() {
    let base = json!({"model":"minimax-h3","operation":"image_to_video","prompt":"synthetic","width":384,"height":256,"seed":42,
 "sampling":{"steps":20,"cfg_scale":1.,"sampler":"res_multistep","scheduler":"simple"},"output":{"format":"mp4","count":1},
 "inputs":{"start_image":"asset_owned"},"video":{"profile":"h3-fl2va-i2va-v1","frames":39,"fps":24,"audio":true}});
    let good: JobRequest = serde_json::from_value(base.clone()).unwrap();
    good.validate().unwrap();
    assert_eq!(
        video::job::request(&good).unwrap().profile,
        h3::Profile::ImageToAvV1
    );
    for (field, value) in [
        ("start_image", serde_json::Value::Null),
        ("end_image", json!("asset_end")),
        ("reference_images", json!(["asset_ref"])),
        ("pose_guide", json!("asset_pose")),
        ("mask", json!("asset_mask")),
    ] {
        let mut bad = base.clone();
        bad["inputs"][field] = value;
        assert!(
            serde_json::from_value::<JobRequest>(bad)
                .unwrap()
                .validate()
                .is_err()
        );
    }
    for (field, value) in [
        ("operation", json!("text_to_video")),
        ("width", json!(385)),
        ("seed", json!(1)),
    ] {
        let mut bad = base.clone();
        bad[field] = value;
        assert!(
            serde_json::from_value::<JobRequest>(bad)
                .unwrap()
                .validate()
                .is_err()
        );
    }
    for (field, value) in [
        ("steps", json!(4)),
        ("steps", json!(19)),
        ("steps", json!(21)),
        ("sampler", json!("euler")),
        ("scheduler", json!("normal")),
        ("cfg_scale", json!(2.)),
        ("custom_sigmas", json!([1., 0.])),
    ] {
        let mut bad = base.clone();
        bad["sampling"][field] = value;
        assert!(
            serde_json::from_value::<JobRequest>(bad)
                .unwrap()
                .validate()
                .is_err()
        );
    }
    let mut bad = base;
    bad["video"]["profile"] = "h3-av-noref-v2".into();
    assert!(
        serde_json::from_value::<JobRequest>(bad)
            .unwrap()
            .validate()
            .is_err()
    );
}
#[test]
fn source_ratio_uses_director_area_rounding_and_keeps_portrait_source() {
    for quality in h3::geometry::QUALITIES {
        let source = h3::Selection {
            ratio: "Source".into(),
            quality: quality.into(),
            ..Default::default()
        };
        assert!(source.resolve().is_err());
        assert!(source.resolve_with_source(Some((0, 100))).is_err());
        let normal = h3::Selection {
            ratio: "3:2".into(),
            quality: quality.into(),
            ..Default::default()
        };
        assert_eq!(
            source.resolve_with_source(Some((1500, 1000))).unwrap(),
            normal.resolve().unwrap()
        );
        let portrait = h3::Selection {
            portrait: true,
            ..normal
        };
        assert_eq!(
            source.resolve_with_source(Some((1000, 1500))).unwrap(),
            portrait.resolve().unwrap()
        );
    }
}
#[test]
fn bounded_image_decode_preserves_rgb_and_hash_without_paths() {
    use std::io::Cursor;
    let mut encoded = Cursor::new(Vec::new());
    image::RgbaImage::from_pixel(2, 3, image::Rgba([10, 20, 30, 0]))
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let (d, rgb) = video::image::decode(encoded.get_ref()).unwrap();
    assert_eq!((d.width, d.height), (2, 3));
    assert_eq!(rgb, [10, 20, 30].repeat(6));
    assert_eq!(d.sha256, imagine_server::storage::hash(&rgb));
    assert!(video::image::decode(b"not an image").is_err());
    let mut asset =
        json!({"width":2,"height":3,"mime_type":"image/png","bytes":encoded.get_ref().len()});
    video::image::validate_asset(&asset).unwrap();
    asset["bytes"] = (h3::MAX_INPUT_IMAGE + 1).into();
    assert!(video::image::validate_asset(&asset).is_err());
    asset["bytes"] = 12.into();
    asset["width"] = 9000.into();
    assert!(video::image::validate_asset(&asset).is_err());
}
