use super::{binding_tests::check, kind, kitchen};
use imagine_inference::{
    execution_value::Value as V,
    sampling::{deis, numpy_scalar},
};
use serde_json::Value;
use tch::{Device, Tensor};
#[test]
fn shared_deis_original_time_gradient_and_coefficients_match() {
    let native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-deis.json")).unwrap();
    for case in fixture["log_cases"].as_array().unwrap() {
        use sha2::{Digest, Sha256};
        let mut bytes = Vec::new();
        for i in 0..case["count"].as_u64().unwrap() as u32 {
            let input = i.wrapping_mul(2654435761).wrapping_add(12345);
            bytes.extend(numpy_scalar::log_f32_avx2(f32::from_bits(input)).to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            case["output_sha256"].as_str().unwrap(),
            "NumPy log Float32 AVX2 lane algorithm"
        );
    }
    for c in fixture["log64_cases"].as_array().unwrap() {
        use sha2::{Digest, Sha256};
        let mut logs = Vec::new();
        let mut exps = Vec::new();
        for i in 0..c["count"].as_u64().unwrap() {
            let bits = i
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = f64::from_bits(bits);
            logs.extend(numpy_scalar::log_f64_numpy(x).to_le_bytes());
            exps.extend(x.exp().to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", Sha256::digest(logs)),
            c["log_sha256"].as_str().unwrap(),
            "NumPy Float64 log bits"
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(exps)),
            c["exp_sha256"].as_str().unwrap(),
            "NumPy Float64 exp bits"
        );
    }
    for case in fixture["cases"].as_array().unwrap() {
        let _inference =
            imagine_inference::native::inference_mode(case["inference"].as_bool().unwrap_or(false))
                .unwrap();
        let sigmas = Tensor::from_slice(&[0.9, 0.6, 0.3, 0.1, 0.05])
            .to_kind(kind(case["kind"].as_str().unwrap()));
        let (time, beta0, beta1) = deis::edm_to_time(
            &native,
            &sigmas,
            1e-3,
            deis::Scalar::Float(0.002),
            deis::Scalar::Integer(80),
        )
        .unwrap();
        for (name, tensor) in [("time", &time), ("beta0", &beta0), ("beta1", &beta1)] {
            check(
                &V::tensor(tensor.shallow_clone()),
                &case[name],
                Device::Cpu,
                &format!("DEIS {name} {}", case["kind"]),
            );
        }
        let taus = Tensor::f_linspace_tensor_tensor(
            &time.get(1),
            &time.get(2),
            10000,
            (native.default_dtype().unwrap(), Device::Cpu),
        )
        .unwrap();
        check(
            &V::tensor(taus.shallow_clone()),
            &case["taus"],
            Device::Cpu,
            "DEIS taus",
        );
        check(
            &V::tensor(deis::integrand(&beta0, &beta1, &taus).unwrap()),
            &case["integrand"],
            Device::Cpu,
            "DEIS original backward integrand",
        );
        let coefficients = deis::coefficients(&native, &sigmas, 3, 10000, "tab").unwrap();
        for (row, expected) in coefficients
            .iter()
            .zip(case["coefficients"].as_array().unwrap())
        {
            for (c, expected) in row.iter().zip(expected.as_array().unwrap()) {
                check(
                    &V::tensor(c.shallow_clone()),
                    &expected["output"],
                    Device::Cpu,
                    "DEIS coefficient",
                );
            }
        }
    }
}
