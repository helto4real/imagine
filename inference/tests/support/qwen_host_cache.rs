use super::{binding_tests::check, kitchen};
use imagine_inference::{
    execution::Runtime, execution_value::Value as V, models::qwen_image21 as q,
    triton_compiler::Compiler,
};
use serde_json::Value;
use std::{path::Path, sync::Arc};
use tch::{Device, Kind, Tensor};

#[test]
fn qwen_image21_original_static_host_prefix_pinning_staging_and_budget_match() {
    let _dir = super::triton_file_cache_tests::OwnDirectory::create();
    let native = kitchen();
    let own = std::env::current_dir().unwrap();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    });
    let rt = Runtime::frozen(native, compiler).unwrap();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-host-cache.json")).unwrap();
    for c in fixture["budget"].as_array().unwrap() {
        assert_eq!(
            q::single_owner_pin_budget(
                c["size"].as_u64().unwrap() as usize,
                c["available"].as_u64().unwrap()
            ),
            c["result"].as_bool().unwrap()
        );
    }
    for c in fixture["cgroups"].as_array().unwrap() {
        let groups = c["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| {
                (
                    g[0].as_i64().unwrap() as i128,
                    if g[1].is_null() {
                        None
                    } else {
                        Some((
                            g[1][0].as_i64().unwrap() as i128,
                            g[1][1].as_i64().unwrap() as i128,
                        ))
                    },
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            q::host_available_from_cgroups(
                c["total"].as_u64().unwrap(),
                c["available"].as_u64().unwrap(),
                &groups
            ),
            c["result"].as_u64().unwrap()
        );
    }
    for route in fixture["routes"].as_array().unwrap() {
        let stored = super::kind(route["stored"].as_str().unwrap());
        let target = super::kind(route["target"].as_str().unwrap());
        let dev = Device::Cuda(0);
        let mut cache = q::PrefixCache::new(
            &rt,
            q::CacheOptions {
                device: "cpu".into(),
                ..Default::default()
            },
            q::StaticCacheServices::new(rt.native.clone(), 0).unwrap(),
        );
        assert!(matches!(
            q::PrefixKv::select(
                &mut cache,
                &Tensor::from_slice(&[1_f32]).to_device(dev),
                4 * 1024 * 2 * 128 * stored.elt_size_in_bytes(),
                4,
                dev
            )
            .unwrap(),
            q::CacheSelection::Fill
        ));
        for i in 0..4 {
            let value = Tensor::arange(1024 * 2 * 128, (Kind::Float, dev))
                .g_add_scalar(i + 3)
                .remainder(29)
                .g_mul_scalar(0.03125)
                .g_sub_scalar(0.40625)
                .view([1, 1024, 2, 128])
                .to_kind(stored);
            q::PrefixKv::put(&mut cache, i as usize, value).unwrap();
        }
        assert_eq!(
            cache.services.pinned_bytes() as u64,
            route["pinned"].as_u64().unwrap()
        );
        assert_eq!(
            cache.memory_bytes() as u64,
            route["memory"].as_u64().unwrap()
        );
        let mut values = Vec::new();
        for i in 0..4 {
            q::PrefixKv::prefetch(&mut cache, i, dev, target).unwrap();
            let value = q::PrefixKv::take(&mut cache, i, dev, target, 1).unwrap();
            check(
                &V::tensor(value.shallow_clone()),
                &route["outputs"][i],
                dev,
                "original pinned host staging",
            );
            values.push(value);
        }
        assert_eq!(
            cache.services.staging_buffers() as u64,
            route["pair_buffers"].as_u64().unwrap()
        );
        assert_eq!(
            values[0].data_ptr() == values[2].data_ptr(),
            route["reused_zero_two"].as_bool().unwrap()
        );
        assert_eq!(
            values[1].data_ptr() == values[3].data_ptr(),
            route["reused_one_three"].as_bool().unwrap()
        );
        check(
            &V::tensor(values[0].shallow_clone()),
            &route["retained_zero_after_reuse"],
            dev,
            "original staging alias lifetime",
        );
        q::PrefixKv::free(&mut cache).unwrap();
        assert_eq!(cache.services.pinned_bytes(), 0);
        assert_eq!(cache.services.staging_buffers(), 0);
    }
}
