//! Executable registered kernel bodies with their original default options.
//! The supplied Execution owns source model/extra_args/callback/noise bindings.
//! DDIM is constructed through KSAMPLER's random-inpaint alias, not a raw kernel.
use super::{
    adams, deis, exponential,
    factory::DpmWrapper,
    parameterization::Parameterization,
    sde,
    solvers::{self, Execution},
    unipc,
};
use crate::{
    models::qwen_image21::EulerSettings,
    native::{Error, Kitchen},
};
use tch::Tensor;

pub fn sample(
    name: &str,
    exec: &mut dyn Execution,
    native: &Kitchen,
    x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    warning: &mut dyn FnMut(&str),
) -> Result<Tensor, Error> {
    match name {
        "euler" | "heun" | "heunpp2" | "dpm_2" => solvers::churn(
            exec,
            x,
            sigmas,
            EulerSettings::default(),
            match name {
                "euler" => solvers::ChurnSolver::Euler,
                "heun" => solvers::ChurnSolver::Heun,
                "heunpp2" => solvers::ChurnSolver::HeunPP2,
                _ => solvers::ChurnSolver::Dpm2,
            },
        ),
        "euler_ancestral" | "dpm_2_ancestral" => solvers::sample_ancestral(
            exec,
            x,
            sigmas,
            model,
            1.,
            1.,
            if name == "euler_ancestral" {
                solvers::AncestralSolver::Euler
            } else {
                solvers::AncestralSolver::Dpm2
            },
        ),
        "dpmpp_2m" | "dpmpp_2m_cfg_pp" => {
            solvers::dpmpp_2m(exec, x, sigmas, name.ends_with("cfg_pp"))
        }
        "gradient_estimation" | "gradient_estimation_cfg_pp" => {
            solvers::gradient_estimation(exec, x, sigmas, 2., name.ends_with("cfg_pp"))
        }
        "euler_cfg_pp" | "euler_ancestral_cfg_pp" => solvers::euler_cfg_pp(
            exec,
            x,
            sigmas,
            model,
            if name == "euler_cfg_pp" { 0. } else { 1. },
            if name == "euler_cfg_pp" { 0. } else { 1. },
        ),
        "dpmpp_2s_ancestral" | "dpmpp_2s_ancestral_cfg_pp" => {
            solvers::dpmpp_2s_ancestral(exec, x, sigmas, model, 1., 1., name.ends_with("cfg_pp"))
        }
        "res_multistep"
        | "res_multistep_cfg_pp"
        | "res_multistep_ancestral"
        | "res_multistep_ancestral_cfg_pp" => solvers::res_multistep(
            exec,
            x,
            sigmas,
            model,
            if name.contains("ancestral") { 1. } else { 0. },
            1.,
            name.ends_with("cfg_pp"),
        ),
        "ddpm" => solvers::ddpm(exec, x, sigmas),
        "lcm" => solvers::lcm(exec, x, sigmas, model, Default::default()),
        "ipndm" => solvers::ipndm(exec, x, sigmas, 4),
        "ipndm_v" => solvers::ipndm_v(exec, x, sigmas, 4),
        "lms" => solvers::lms(exec, x, sigmas, 4, warning),
        "cfgpp_ud10_ab" => solvers::cfgpp_ud10_ab(exec, x, sigmas, model, warning),
        "deis" => deis::sample(exec, native, x, sigmas, 3, "tab"),
        "uni_pc" => unipc::sample(exec, native, x, sigmas, "bh1"),
        "uni_pc_bh2" => unipc::sample_bh2(exec, native, x, sigmas),
        "sa_solver" | "sa_solver_pece" => adams::sample(
            exec,
            x,
            sigmas,
            model,
            adams::Settings {
                pece: name.ends_with("pece"),
                ..Default::default()
            },
            None,
        ),
        "dpm_fast" => DpmWrapper::Fast.sample_defaults(exec, native, x, sigmas),
        "dpm_adaptive" => DpmWrapper::Adaptive.sample_defaults(exec, native, x, sigmas),
        "seeds_2" => exponential::seeds_2(exec, x, sigmas, model, Default::default(), "phi_1"),
        "seeds_3" => exponential::seeds_3(exec, x, sigmas, model, Default::default()),
        "exp_heun_2_x0" | "exp_heun_2_x0_sde" => exponential::seeds_2(
            exec,
            x,
            sigmas,
            model,
            exponential::Seeds2 {
                eta: if name == "exp_heun_2_x0" { 0. } else { 1. },
                noise: if name == "exp_heun_2_x0" { 0. } else { 1. },
                r: 1.,
            },
            "phi_2",
        ),
        "er_sde" => exponential::er_sde(exec, x, sigmas, model, 1., 3, None),
        "dpmpp_sde"
        | "dpmpp_sde_gpu"
        | "dpmpp_2m_sde"
        | "dpmpp_2m_sde_gpu"
        | "dpmpp_2m_sde_heun"
        | "dpmpp_2m_sde_heun_gpu"
        | "dpmpp_3m_sde"
        | "dpmpp_3m_sde_gpu" => {
            let settings = sde::Settings {
                source: if name.ends_with("_gpu") {
                    sde::Source::Gpu
                } else {
                    sde::Source::Cpu
                },
                ..Default::default()
            };
            if name.starts_with("dpmpp_2m") {
                sde::dpmpp_2m_sde(
                    exec,
                    x,
                    sigmas,
                    model,
                    settings,
                    if name.contains("heun") {
                        "heun"
                    } else {
                        "midpoint"
                    },
                )
            } else if name.starts_with("dpmpp_3m") {
                sde::dpmpp_3m_sde(exec, x, sigmas, model, settings)
            } else {
                sde::dpmpp_sde(exec, x, sigmas, model, settings, 0.5)
            }
        }
        _ => Err(Error::BindingGap {
            backend: "registered_sampler_defaults".into(),
            operation: if name == "ddim" {
                "DDIM requires the source KSAMPLER random-inpaint binding".into()
            } else {
                format!("unregistered kernel {name}")
            },
        }),
    }
}
