// Generated from locked comfy/k_diffusion/sampling.py::sample_ipndm_v.
// IPNDM source attribution: zju-pi/diff-sampler, Apache-2.0 (as in reference).
// Source SHA256: cfc4221ed0bc1f84a6f5689d3ec440cad3008c2541ab8f5e2599d1a73fac74ce
use super::locals::SourceLocals;
use crate::native::Error;
use tch::Tensor;
pub(super) fn coefficients(
    frame: &mut SourceLocals,
    sigmas: &Tensor,
    i: i64,
    order: i64,
) -> Result<Vec<Tensor>, Error> {
    let t_cur = sigmas.f_get(i)?;
    let t_next = sigmas.f_get(i + 1)?;
    match order {
        2 => {
            let h_n = frame.keep("h_n", (t_next).f_sub(&(t_cur))?);
            let h_n_1 = frame.keep("h_n_1", (t_cur).f_sub(&(sigmas.f_get(i - 1)?))?);
            let coeff1 = frame.keep(
                "coeff1",
                (((h_n).f_div(&(h_n_1))?).f_add_scalar(2)?).f_div_scalar(2)?,
            );
            let coeff2 = frame.keep(
                "coeff2",
                (((h_n).f_div(&(h_n_1))?).f_neg()?).f_div_scalar(2)?,
            );
            Ok(vec![coeff1, coeff2])
        }
        3 => {
            let h_n = frame.keep("h_n", (t_next).f_sub(&(t_cur))?);
            let h_n_1 = frame.keep("h_n_1", (t_cur).f_sub(&(sigmas.f_get(i - 1)?))?);
            let h_n_2 = frame.keep(
                "h_n_2",
                (sigmas.f_get(i - 1)?).f_sub(&(sigmas.f_get(i - 2)?))?,
            );
            let temp = frame.keep(
                "temp",
                (((((h_n).f_div(&(((h_n).f_add(&(h_n_1))?).f_mul_scalar(3)?))?)
                    .f_mul(&((h_n).f_mul(&((h_n).f_add(&(h_n_1))?))?))?)
                .f_div(&((h_n_1).f_mul(&((h_n_1).f_add(&(h_n_2))?))?))?)
                .f_rsub_scalar(1)?)
                .f_div_scalar(2)?,
            );
            let coeff1 = frame.keep(
                "coeff1",
                ((((h_n).f_div(&(h_n_1))?).f_add_scalar(2)?).f_div_scalar(2)?).f_add(&(temp))?,
            );
            let coeff2 = frame.keep(
                "coeff2",
                ((((h_n).f_div(&(h_n_1))?).f_neg()?).f_div_scalar(2)?)
                    .f_sub(&((((h_n_1).f_div(&(h_n_2))?).f_add_scalar(1)?).f_mul(&(temp))?))?,
            );
            let coeff3 = frame.keep("coeff3", ((temp).f_mul(&(h_n_1))?).f_div(&(h_n_2))?);
            Ok(vec![coeff1, coeff2, coeff3])
        }
        4 => {
            let h_n = frame.keep("h_n", (t_next).f_sub(&(t_cur))?);
            let h_n_1 = frame.keep("h_n_1", (t_cur).f_sub(&(sigmas.f_get(i - 1)?))?);
            let h_n_2 = frame.keep(
                "h_n_2",
                (sigmas.f_get(i - 1)?).f_sub(&(sigmas.f_get(i - 2)?))?,
            );
            let h_n_3 = frame.keep(
                "h_n_3",
                (sigmas.f_get(i - 2)?).f_sub(&(sigmas.f_get(i - 3)?))?,
            );
            let temp1 = frame.keep(
                "temp1",
                (((((h_n).f_div(&(((h_n).f_add(&(h_n_1))?).f_mul_scalar(3)?))?)
                    .f_mul(&((h_n).f_mul(&((h_n).f_add(&(h_n_1))?))?))?)
                .f_div(&((h_n_1).f_mul(&((h_n_1).f_add(&(h_n_2))?))?))?)
                .f_rsub_scalar(1)?)
                .f_div_scalar(2)?,
            );
            let temp2 = frame.keep(
                "temp2",
                ((((((h_n).f_div(&(((h_n).f_add(&(h_n_1))?).f_mul_scalar(3)?))?)
                    .f_rsub_scalar(1)?)
                .f_div_scalar(2)?)
                .f_add(
                    &(((((h_n).f_div(&(((h_n).f_add(&(h_n_1))?).f_mul_scalar(2)?))?)
                        .f_rsub_scalar(1)?)
                    .f_mul(&(h_n))?)
                    .f_div(&((((h_n).f_add(&(h_n_1))?).f_add(&(h_n_2))?).f_mul_scalar(6)?))?),
                )?)
                .f_mul(
                    &(((h_n).f_mul(&((h_n).f_add(&(h_n_1))?))?)
                        .f_mul(&(((h_n).f_add(&(h_n_1))?).f_add(&(h_n_2))?))?),
                )?)
                .f_div(
                    &(((h_n_1).f_mul(&((h_n_1).f_add(&(h_n_2))?))?)
                        .f_mul(&(((h_n_1).f_add(&(h_n_2))?).f_add(&(h_n_3))?))?),
                )?,
            );
            let coeff1 = frame.keep(
                "coeff1",
                (((((h_n).f_div(&(h_n_1))?).f_add_scalar(2)?).f_div_scalar(2)?).f_add(&(temp1))?)
                    .f_add(&(temp2))?,
            );
            let coeff2 = frame.keep(
                "coeff2",
                (((((h_n).f_div(&(h_n_1))?).f_neg()?).f_div_scalar(2)?)
                    .f_sub(&((((h_n_1).f_div(&(h_n_2))?).f_add_scalar(1)?).f_mul(&(temp1))?))?)
                .f_sub(
                    &(((((h_n_1).f_div(&(h_n_2))?).f_add_scalar(1)?).f_add(
                        &(((h_n_1).f_mul(&((h_n_1).f_add(&(h_n_2))?))?)
                            .f_div(&((h_n_2).f_mul(&((h_n_2).f_add(&(h_n_3))?))?))?),
                    )?)
                    .f_mul(&(temp2))?),
                )?,
            );
            let coeff3 = frame.keep(
                "coeff3",
                (((temp1).f_mul(&(h_n_1))?).f_div(&(h_n_2))?).f_add(
                    &((((h_n_1).f_div(&(h_n_2))?).f_add(
                        &((((h_n_1).f_mul(&((h_n_1).f_add(&(h_n_2))?))?)
                            .f_div(&((h_n_2).f_mul(&((h_n_2).f_add(&(h_n_3))?))?))?)
                        .f_mul(&(((h_n_2).f_div(&(h_n_3))?).f_add_scalar(1)?))?),
                    )?)
                    .f_mul(&(temp2))?),
                )?,
            );
            let coeff4 = frame.keep(
                "coeff4",
                ((((temp2).f_neg()?).f_mul(
                    &(((h_n_1).f_mul(&((h_n_1).f_add(&(h_n_2))?))?)
                        .f_div(&((h_n_2).f_mul(&((h_n_2).f_add(&(h_n_3))?))?))?),
                )?)
                .f_mul(&(h_n_1))?)
                .f_div(&(h_n_2))?,
            );
            Ok(vec![coeff1, coeff2, coeff3, coeff4])
        }
        _ => Ok(Vec::new()),
    }
}
