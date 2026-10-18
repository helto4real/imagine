//! Original diffusionmodules.model pytorch_attention and its slice/OOM retry.
use crate::models::qwen_image21::{host_available_memory, image_vae_is_oom};
use crate::{
    execution::Runtime,
    native::{Error, SdpaOptions},
};
use tch::{Device, Tensor};
pub trait FluxVaeAttentionServices {
    fn sdpa(&mut self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, Error>;
    fn bmm(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
        Ok(a.f_bmm(b)?)
    }
    fn free_memory(&mut self, device: Device) -> Result<f64, Error>;
    fn is_oom(&mut self, device: Device, error: &Error) -> Result<bool, Error>;
    fn empty_cache(&mut self, device: Device) -> Result<(), Error>;
    fn warning(&mut self, message: &str) {
        eprintln!("WARNING: {message}");
    }
}
struct Services<'a>(&'a Runtime);
impl FluxVaeAttentionServices for Services<'_> {
    fn sdpa(&mut self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, Error> {
        self.0.native.comfy_sdpa(
            q,
            k,
            v,
            SdpaOptions {
                mask: None,
                scale: None,
                enable_gqa: false,
                priority_wrapper: true,
            },
        )
    }
    fn free_memory(&mut self, device: Device) -> Result<f64, Error> {
        match device {
            Device::Cpu => Ok(host_available_memory()? as f64),
            Device::Cuda(i) => {
                let [free, _, active, reserved] = self.0.native.cuda_memory(i)?;
                Ok((free + reserved - active) as f64)
            }
            _ => Err(Error::BindingGap {
                backend: "flux2_vae".into(),
                operation: "Non-initial-platform attention memory".into(),
            }),
        }
    }
    fn is_oom(&mut self, device: Device, error: &Error) -> Result<bool, Error> {
        Ok(image_vae_is_oom(&self.0.native, device, error))
    }
    fn empty_cache(&mut self, device: Device) -> Result<(), Error> {
        if let Device::Cuda(i) = device {
            self.0.native.cuda_soft_empty_cache(i)?;
        }
        Ok(())
    }
}
pub fn moss_vae_attention(
    rt: &Runtime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Tensor, Error> {
    flux2_vae_attention_with_services(q, k, v, &mut Services(rt))
}
pub fn flux2_vae_attention_with_services<S: FluxVaeAttentionServices>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    s: &mut S,
) -> Result<Tensor, Error> {
    let original = q.size();
    let (b, c) = (original[0], original[1]);
    // Source views and contiguous copies occur before its try/except.
    let shape = |x: &Tensor| x.f_view([b, 1, c, -1])?.f_transpose(2, 3)?.f_contiguous();
    let q = shape(q)?;
    let k = shape(k)?;
    let v = shape(v)?;
    let attempt = s
        .sdpa(&q, &k, &v)
        .and_then(|out| Ok(out.f_transpose(2, 3)?.f_reshape(original.as_slice())?));
    match attempt {
        Ok(out) => Ok(out),
        Err(error) => {
            if !s.is_oom(q.device(), &error)? {
                return Err(error);
            }
            drop(error);
            s.warning("scaled_dot_product_attention OOMed: switched to slice attention");
            let q = q.f_view([b, -1, c])?;
            let k = k.f_view([b, -1, c])?.f_transpose(1, 2)?;
            let v = v.f_view([b, -1, c])?.f_transpose(1, 2)?;
            let out = k.f_zeros_like()?;
            let scale = (q.size()[2] as f64).powf(-0.5);
            let free = s.free_memory(q.device())?;
            let required =
                (q.size()[0] * q.size()[1] * k.size()[2] * q.kind().elt_size_in_bytes() as i64)
                    as f64
                    * if q.kind().elt_size_in_bytes() == 2 {
                        3.
                    } else {
                        2.5
                    };
            let mut steps = 1_i64;
            if required > free {
                if free == 0. {
                    return Err(Error::ReferenceException {
                        class: "ZeroDivisionError",
                        message: if q.kind().elt_size_in_bytes() == 2 {
                            "division by zero"
                        } else {
                            "float division by zero"
                        }
                        .into(),
                    });
                }
                steps = 2_i64.pow(((required / free).ln() / 2_f64.ln()).ceil() as u32);
            }
            let mut logits: Option<Tensor> = None;
            let mut probabilities: Option<Tensor> = None;
            loop {
                let attempt =
                    (|| -> Result<(), Error> {
                        let size = if q.size()[1] % steps == 0 {
                            q.size()[1] / steps
                        } else {
                            q.size()[1]
                        };
                        if size == 0 {
                            return Err(Error::ReferenceException {
                                class: "ValueError",
                                message: "range() arg 3 must not be zero".into(),
                            });
                        }
                        let mut i = 0;
                        while i < q.size()[1] {
                            let end = (i + size).min(q.size()[1]);
                            drop(logits.replace(
                                s.bmm(&q.f_slice(1, i, end, 1)?, &k)?.f_mul_scalar(scale)?,
                            ));
                            drop(
                                probabilities.replace(
                                    logits
                                        .as_ref()
                                        .unwrap()
                                        .f_softmax(2, None)?
                                        .f_permute([0, 2, 1])?,
                                ),
                            );
                            drop(logits.take());
                            out.f_slice(2, i, end, 1)?
                                .f_copy_(&s.bmm(&v, probabilities.as_ref().unwrap())?)?;
                            drop(probabilities.take());
                            i += size;
                        }
                        Ok(())
                    })();
                match attempt {
                    Ok(()) => break,
                    Err(error) => {
                        if !s.is_oom(q.device(), &error)? {
                            return Err(error);
                        }
                        s.empty_cache(q.device())?;
                        steps *= 2;
                        if steps > 128 {
                            return Err(error);
                        }
                        s.warning(&format!(
                            "out of memory error, increasing steps and trying again {steps}"
                        ));
                    }
                }
            }
            Ok(out.f_reshape(original.as_slice())?)
        }
    }
}
