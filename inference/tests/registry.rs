use imagine_inference::registry::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use tch::Kind;

fn registry() -> BackendRegistry {
    let registry = BackendRegistry::new(Some((8, 9)));
    let cuda = FunctionConstraints {
        params: vec![(
            "x".into(),
            ParamConstraint {
                types: Some(vec![AllowedType::DType(Kind::Float)]),
                shape_rules: vec![
                    ShapeRule::ExactDims(2),
                    ShapeRule::DivisibleBy { dim: 1, factor: 16 },
                ],
                ..Default::default()
            },
        )],
        default_devices: BTreeSet::from(["cuda".into()]),
        min_capability: Some((8, 0)),
        ..Default::default()
    };
    registry.register("cuda", BTreeMap::from([("quantize".into(), cuda)]));
    registry.register(
        "eager",
        BTreeMap::from([("quantize".into(), FunctionConstraints::default())]),
    );
    registry
}
fn arguments(device: &str, shape: Vec<i64>) -> CallArgs<'static> {
    BTreeMap::from([(
        "x".into(),
        Argument::Tensor {
            kind: Kind::Float,
            device: device.into(),
            shape,
        },
    )])
}

#[test]
fn selection_preserves_override_fallback_and_explicit_failure() {
    let registry = registry();
    let gpu = arguments("cuda", vec![4, 16]);
    let cpu = arguments("cpu", vec![4, 16]);
    assert_eq!(registry.select("quantize", None, &gpu).unwrap(), "cuda");
    assert_eq!(registry.select("quantize", None, &cpu).unwrap(), "eager");
    assert!(matches!(
        registry.select("quantize", Some("cuda"), &cpu),
        Err(SelectionError::NoCapableBackend { .. })
    ));
    let outer = registry.use_backend("eager").unwrap();
    assert_eq!(registry.select("quantize", None, &gpu).unwrap(), "eager");
    {
        let _inner = registry.use_backend("cuda").unwrap();
        assert_eq!(registry.select("quantize", None, &gpu).unwrap(), "cuda");
        assert_eq!(registry.select("quantize", None, &cpu).unwrap(), "eager");
        assert_eq!(
            registry
                .select("quantize", None, &arguments("cuda", vec![4, 17]))
                .unwrap(),
            "eager"
        );
    }
    assert_eq!(registry.select("quantize", None, &gpu).unwrap(), "eager");
    drop(outer);
    assert_eq!(registry.select("quantize", None, &gpu).unwrap(), "cuda");
    registry.disable("cuda");
    assert!(!registry.is_available("cuda"));
    assert!(registry.list_backends()["cuda"].available);
    assert!(matches!(
        registry.select("quantize", Some("cuda"), &gpu),
        Err(SelectionError::BackendNotFound { .. })
    ));
    assert_eq!(registry.select("quantize", None, &gpu).unwrap(), "eager");
    registry.enable("cuda");
    registry.mark_unavailable("cuda", "recorded reason");
    // mark_unavailable alone does not unregister an already registered backend.
    assert!(registry.is_available("cuda"));
    assert_eq!(registry.select("quantize", None, &gpu).unwrap(), "cuda");
}

#[test]
fn override_is_thread_local_and_restores_after_unwind() {
    let registry = registry();
    let _outer = registry.use_backend("eager").unwrap();
    std::thread::scope(|scope| {
        let borrowed = &registry;
        scope
            .spawn(move || {
                assert_eq!(
                    borrowed
                        .select("quantize", None, &arguments("cuda", vec![4, 16]))
                        .unwrap(),
                    "cuda"
                )
            })
            .join()
            .unwrap();
    });
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _inner = registry.use_backend("cuda").unwrap();
        panic!("synthetic unwinding");
    }));
    assert_eq!(
        registry
            .select("quantize", None, &arguments("cuda", vec![4, 16]))
            .unwrap(),
        "eager"
    );
}

#[test]
fn missing_arguments_hardware_and_call_rules_follow_reference_order() {
    let registry = registry();
    // Empty kwargs skip validation, and missing constrained kwargs mean None.
    assert_eq!(
        registry.select("quantize", None, &CallArgs::new()).unwrap(),
        "cuda"
    );
    assert_eq!(
        registry
            .select(
                "quantize",
                None,
                &BTreeMap::from([("other".into(), Argument::None)])
            )
            .unwrap(),
        "cuda"
    );
    assert!(matches!(
        registry.select("missing", Some("cuda"), &CallArgs::new()),
        Err(SelectionError::BackendNotImplemented { .. })
    ));
    let constraints = FunctionConstraints {
        min_capability: Some((9, 0)),
        call_rules: vec![Arc::new(|_| {
            Err(Failure::new("call", "must not run first"))
        })],
        ..Default::default()
    };
    assert_eq!(
        constraints
            .validate(&CallArgs::new(), Some((8, 9)))
            .unwrap_err()
            .parameter,
        "__hardware__"
    );
    assert_eq!(
        constraints
            .validate(&CallArgs::new(), Some((9, 0)))
            .unwrap_err()
            .parameter,
        "call"
    );
    registry.register("cuda", BTreeMap::new());
    assert!(registry.get_constraints("cuda", "quantize").is_some());
    assert!(matches!(
        registry.select("quantize", Some("cuda"), &CallArgs::new()),
        Err(SelectionError::BackendNotImplemented { .. })
    ));
}

#[test]
fn hardware_probe_is_lazy_cached_and_rule_exceptions_never_fall_back() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let registry = BackendRegistry::with_capability_probe(move || {
        observed.fetch_add(1, Ordering::SeqCst);
        Some((8, 9))
    });
    registry.register(
        "cuda",
        BTreeMap::from([("operation".into(), FunctionConstraints::default())]),
    );
    registry
        .select("operation", None, &CallArgs::new())
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    registry
        .validate_backend_for_call("cuda", "operation", &CallArgs::new())
        .unwrap();
    registry
        .clone()
        .select("operation", None, &arguments("cuda", vec![1]))
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    registry.register(
        "cuda",
        BTreeMap::from([(
            "operation".into(),
            FunctionConstraints {
                call_rules: vec![Arc::new(|_| Err(Failure::raised("KeyError", "'x'")))],
                ..Default::default()
            },
        )]),
    );
    registry.register(
        "eager",
        BTreeMap::from([("operation".into(), FunctionConstraints::default())]),
    );
    assert!(matches!(
        registry.select("operation", None, &arguments("cuda", vec![1])),
        Err(SelectionError::CallRuleRaised(_))
    ));
    registry.disable("cuda");
    registry.mark_unavailable("cuda", "original extension error");
    assert_eq!(
        registry
            .validate_backend_for_call("cuda", "operation", &CallArgs::new())
            .unwrap_err(),
        Failure::new("__backend__", "original extension error")
    );
    // get_implementation checks registration before disabled state; use_backend
    // prioritizes a recorded unavailable reason. They intentionally differ.
    registry.disable("missing");
    registry.mark_unavailable("missing", "load error");
    assert!(
        matches!(registry.select("operation",Some("missing"),&CallArgs::new()),Err(SelectionError::BackendNotFound{reason,..}) if reason=="load error")
    );
}
