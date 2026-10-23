//! Own text/vision component isolation; metadata only, outside the worker contract.
use imagine_inference::{
    attention::{Attention, AttentionInput, AttentionMethod},
    checkpoint::{
        Aimdo, bind_comfy_layouts,
        component::{self, StorageAnnotations},
    },
    execution::Runtime,
    execution_value::Value,
    memory::Manager,
    models::{minimax_h3 as h3, qwen_image21 as q},
    native::{Error, Kitchen},
    patching::options::Options,
};
use q::TextEncoderExecution;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use tch::{Device, Tensor};
struct Exec {
    native: Kitchen,
    attention: Attention,
    rows: Vec<serde_json::Value>,
}
impl Exec {
    fn mark(&mut self, name: &str, value: Option<&Tensor>) -> Result<(), Error> {
        self.native.synchronize_device(0)?;
        self.rows.push(serde_json::json!({"stage":name,"shape":value.map(Tensor::size),"strides":value.map(Tensor::stride),"dtype":value.map(|v|format!("{:?}",v.kind())),"cuda_memory_free_total_active_reserved":self.native.cuda_memory(0)?}));
        Ok(())
    }
}
impl TextEncoderExecution for Exec {
    fn queue_begin(&mut self, _: usize, _: Device) -> Result<(), Error> {
        self.mark("language.begin", None)
    }
    fn queue_end(&mut self, _: Device) -> Result<(), Error> {
        self.mark("language.end", None)
    }
    fn block(
        &mut self,
        i: usize,
        f: &mut dyn FnMut(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.mark(&format!("language.block.{i}.begin"), None)?;
        f(self)
    }
    fn observe(&mut self, name: &str, value: &Tensor) -> Result<(), Error> {
        self.mark(name, Some(value))
    }
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        heads: i64,
        gqa: bool,
    ) -> Result<Tensor, Error> {
        self.mark("attention.begin", Some(q))?;
        let out = unsafe {
            self.attention.run(
                rt,
                AttentionInput {
                    q,
                    k,
                    v,
                    heads,
                    mask,
                    skip_reshape: true,
                    skip_output_reshape: false,
                    enable_gqa: gqa,
                    scale: None,
                },
            )
        }?;
        self.mark("attention.end", Some(&out))?;
        Ok(out)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if !(args.len() == 5 || args.len() == 6 && args[5] == Path::new("dynamic")) {
        return Err("model own-input own-new-output kitchen aimdo [dynamic]".into());
    }
    let own = Path::new(
        "/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01",
    );
    if !args[1].starts_with(own) || !args[2].starts_with(own) || args[2].exists() {
        return Err("Own encrypted input/new output required".into());
    }
    std::fs::create_dir(&args[2])?;
    tch::set_num_threads(2);
    let _grad = tch::no_grad_guard();
    let rt = Runtime::frozen_partial(unsafe { Kitchen::load(&args[3]) }?)?;
    bind_comfy_layouts(&rt);
    rt.backends.policy.disable("triton");
    let mut exec = Exec {
        native: rt.native.clone(),
        attention: Attention {
            method: AttentionMethod::Pytorch,
            priority_wrapper: true,
            attention_precision: None,
            low_precision_attention: true,
            model_device: Device::Cuda(0),
        },
        rows: vec![],
    };
    let aimdo = unsafe { Aimdo::load(&args[4]) }?;
    let result = (|| -> Result<(), Error> {
        exec.mark("initial", None)?;
        let input = Tensor::read_safetensors(&args[1])?;
        let image = &input.iter().find(|(k, _)| k == "image").unwrap().1;
        let state = unsafe {
            component::load_aimdo(
                &rt,
                &aimdo,
                &args[0],
                &Options::Atom(Value::None),
                &StorageAnnotations::default(),
            )
        }?;
        let parameters =
            h3::checkpoint::prepare_normalized(&rt, h3::checkpoint::Component::Text, &state)?;
        if args.len() == 6 {
            let params = imagine_inference::checkpoint::dynamic::prepare_text(
                &rt,
                &args[4],
                parameters,
                h3::checkpoint::component_inventory(h3::checkpoint::Component::Text)?,
            )?;
            exec.mark("model.loaded.dynamic", None)?;
            let encoder = h3::text_encoder::TextEncoder::from_parameters(&params, true)?;
            let mut tokenizer = h3::tokenizer::Tokenizer::frozen()?;
            let image = Value::tensor(image.shallow_clone());
            let tokens = tokenizer.tokenize(
                "An adult person smiles and waves gently. Audio: quiet ambience.",
                &[],
                &[h3::tokenizer::Reference::Image(&image)],
                false,
            )?;
            let encoding =
                unsafe { encoder.encode(&rt, &tokens, Device::Cuda(0), Device::Cpu, &mut exec) };
            let sync = rt.native.synchronize_device(0);
            if let Ok(value) = &encoding {
                Tensor::write_safetensors(
                    &[("context", &value.context), ("tags", &value.token_tags)],
                    args[2].join("conditioning.safetensors"),
                )?;
            }
            drop(encoder);
            drop(params);
            encoding?;
            sync?;
            return Ok(());
        }
        let pins = q::HostPins::new(rt.native.clone())?;
        let owner = h3::checkpoint::static_component(
            &rt,
            h3::checkpoint::Component::Text,
            parameters,
            BTreeMap::new(),
        )?
        .with_shared_pins(pins.clone());
        let model = q::StaticModel::new(owner, &rt, None);
        let patcher = model.patcher();
        let mut manager = Manager::new();
        let mut memory = q::StaticMemoryServices::new(rt.native.clone(), pins);
        manager.load(
            std::slice::from_ref(&patcher),
            0.,
            None,
            false,
            false,
            &mut memory,
        )?;
        exec.mark("model.loaded", None)?;
        let params = model.owner.borrow().execution_parameters();
        let encoder = h3::text_encoder::TextEncoder::from_parameters(&params, true)?;
        let mut tokenizer = h3::tokenizer::Tokenizer::frozen()?;
        let image = Value::tensor(image.shallow_clone());
        let tokens = tokenizer.tokenize(
            "An adult person smiles and waves gently. Audio: quiet ambience.",
            &[],
            &[h3::tokenizer::Reference::Image(&image)],
            false,
        )?;
        let encoding =
            unsafe { encoder.encode(&rt, &tokens, Device::Cuda(0), Device::Cpu, &mut exec) };
        if let Ok(value) = &encoding {
            Tensor::write_safetensors(
                &[("context", &value.context), ("tags", &value.token_tags)],
                args[2].join("conditioning.safetensors"),
            )?;
        }
        let cleanup = manager.unload_all(Device::Cuda(0), &mut memory);
        encoding?;
        cleanup?;
        Ok(())
    })();
    let error = result.err().map(|e| {
        let text = format!("{e:?}");
        if let Some(start) = text.find("Tried to allocate ") {
            let allocation = &text[start..];
            format!(
                "OutOfMemoryError: {}",
                allocation
                    .split(". GPU")
                    .next()
                    .unwrap_or("allocation failed")
            )
        } else {
            text
        }
    });
    std::fs::write(
        args[2].join("result.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"passed":error.is_none(),"error":error,"trace":exec.rows,"scope":"same static text owner/load policy and source text/vision forwards; metadata only; no full inference"}),
        )?,
    )?;
    Ok(())
}
