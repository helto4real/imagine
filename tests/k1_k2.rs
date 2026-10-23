//! CPU-only regression of image numeric admission and the real H3 deployment module.
use imagine_server::{
    contract::{self, JobRequest},
    h3_protocol as protocol,
};
#[allow(dead_code)]
#[path = "../inference/src/h3_product/deployment.rs"]
mod deployment;
#[path = "../inference/src/server/sampling_values.rs"]
mod sampling_values;

#[test]
fn api_and_worker_accept_larger_finite_sampling_without_losing_numeric_checks() {
    let meta = contract::model(contract::MODEL).unwrap();
    let mut request: JobRequest = serde_json::from_value(serde_json::json!({"model":contract::MODEL,"operation":"text_to_image","prompt":"own fixture"})).unwrap();
    request.prompt = "own synthetic K2 request".into();
    request.sampling.steps = 130;
    request.sampling.cfg_scale = 20.1;
    request.validate().unwrap();
    assert!(sampling_values::valid(130, 20.1));
    assert!(sampling_values::valid(130, -1.));
    request.sampling.cfg_scale = -1.;
    request.validate().unwrap();
    assert!(meta["constraints"]["steps"].get("max").is_none());
    assert!(meta["constraints"]["cfg_scale"].get("max").is_none());
    request.sampling.steps = 0;
    assert_eq!(request.validate(), Err("unsupported_sampling"));
    for cfg in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!sampling_values::valid(130, cfg));
    }
    assert!(!sampling_values::valid(0, 1.));
    assert!(!sampling_values::valid(-1, 1.));
    request.sampling.steps = u32::MAX;
    request.sampling.cfg_scale = 1.;
    request.validate().unwrap();
    request.sampling.cfg_scale = f32::NAN;
    assert_eq!(request.validate(), Err("unsupported_sampling"));
    request.sampling.cfg_scale = 1.;
    request.sampling.sampler = "not-an-implemented-sampler".into();
    assert_eq!(request.validate(), Err("unsupported_sampling"));
}

#[test]
fn explicit_h3_build_audit_still_hashes_bytes_but_declaration_does_not_pin_old_builds() {
    let root = tempfile::tempdir().unwrap();
    let exe = root.path().join("worker");
    let dep = root.path().join("own-library");
    std::fs::write(&exe, b"own executable").unwrap();
    std::fs::write(&dep, b"own dependency").unwrap();
    let dependencies = std::collections::BTreeMap::from([(
        dep.to_str().unwrap().to_owned(),
        deployment::sha(&dep).unwrap(),
    )]);
    let mut identity = deployment::identity(&exe, &dependencies, &[], &[]).unwrap();
    identity.core_revision = "own-compatible-rebuild".into();
    identity.model_lock_sha256 = "c".repeat(64);
    assert!(deployment::declaration(&exe, &dependencies, identity.clone(), &[]).is_ok());
    std::fs::write(&dep, b"own rebuilt dependency").unwrap();
    assert!(deployment::declaration(&exe, &dependencies, identity.clone(), &[]).is_ok());
    assert!(deployment::identity(&exe, &dependencies, &[], &[]).is_err());
    identity.protocol += 1;
    assert!(deployment::declaration(&exe, &dependencies, identity, &[]).is_err());
}

#[test]
fn actual_worker_environment_accepts_new_bridge_provenance() {
    const CHILD: &str = "IMAGINE_OWN_K1_CHILD";
    if std::env::var_os(CHILD).is_some() {
        assert!(deployment::Config::environment().is_ok());
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let dep = root.path().join("own-library");
    std::fs::write(&dep, b"own rebuilt bridge; never loaded").unwrap();
    let dependencies = std::collections::BTreeMap::from([(
        dep.to_str().unwrap().to_owned(),
        deployment::sha(&dep).unwrap(),
    )]);
    assert_ne!(
        dependencies.values().next().unwrap(),
        deployment::BRIDGE_SHA256
    );
    let exe = std::env::current_exe().unwrap();
    let mut identity = deployment::identity(&exe, &dependencies, &[], &[]).unwrap();
    identity.core_revision = "own-compatible-rebuild".into();
    let map = root.path().join("dependencies.json");
    std::fs::write(&map, serde_json::to_vec(&dependencies).unwrap()).unwrap();
    let status = std::process::Command::new(exe)
        .args([
            "--exact",
            "actual_worker_environment_accepts_new_bridge_provenance",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env(deployment::MAP_ENV, &map)
        .env(deployment::MODEL_ENV, root.path())
        .env("IMAGINE_KITCHEN_LIBRARY", &dep)
        .env("IMAGINE_AIMDO_LIBRARY", &dep)
        .env("IMAGINE_PILLOW_LIBRARY", &dep)
        .env(
            "IMAGINE_H3_DEPLOYMENT",
            serde_json::to_string(&identity).unwrap(),
        )
        .status()
        .unwrap();
    assert!(status.success());
}
