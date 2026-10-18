// Generated native signatures from comfy_kitchen/backends/cuda/_C.abi3.so, SHA256 d19debb2fbc197aa26f73496ca30513386fae424a03e28ee2697afde1fb8f988.
pub(crate) const NA3D_SIGNATURE: &str = "na3d(q: ndarray[device='cuda'], k: ndarray[device='cuda'], v: ndarray[device='cuda'], out: ndarray[device='cuda'], batch: int, t_size: int, h_size: int, w_size: int, num_heads: int, head_dim: int, kt: int, kh: int, kw: int, causal_t: int, causal_h: int, causal_w: int, scale: float, dtype_code: int, stream_ptr: int) -> None";
pub(crate) const NA3D_INVOKED: &str = "PyCapsule, PyCapsule, PyCapsule, PyCapsule, int, int, int, int, int, int, int, int, int, int, int, int, float, int, int";
pub(crate) const QUANTIZE_INT8_ROWWISE_SIGNATURE: &str = "quantize_int8_rowwise(input: ndarray[shape=(*, *), device='cuda'], output: ndarray[dtype=int8, shape=(*, *), device='cuda'], scales: ndarray[dtype=float32, shape=(*, *), device='cuda'], stochastic: bool, seed: int, stream_ptr: int) -> None";
pub(crate) const QUANTIZE_INT8_ROWWISE_INVOKED: &str =
    "PyCapsule, PyCapsule, PyCapsule, bool, int, int";
pub(crate) const QUANTIZE_INT8_ROWWISE_CONVROT_SIGNATURE: &str = "quantize_int8_rowwise_convrot(input: ndarray[shape=(*, *), device='cuda'], output: ndarray[dtype=int8, shape=(*, *), device='cuda'], scales: ndarray[dtype=float32, shape=(*, *), device='cuda'], group_size: int, stochastic: bool, seed: int, stream_ptr: int) -> None";
pub(crate) const QUANTIZE_INT8_ROWWISE_CONVROT_INVOKED: &str =
    "PyCapsule, PyCapsule, PyCapsule, int, bool, int, int";
pub(crate) const QUANTIZE_INT8_ROWWISE_CONVROT64_SIGNATURE: &str = "quantize_int8_rowwise_convrot64(input: ndarray[shape=(*, *), device='cuda'], output: ndarray[dtype=int8, shape=(*, *), device='cuda'], scales: ndarray[dtype=float32, shape=(*, *), device='cuda'], group_size: int, stochastic: bool, act_code: int, seed: int, act_weight: ndarray[device='cuda'], act_eps: float, stream_ptr: int) -> None";
pub(crate) const QUANTIZE_INT8_ROWWISE_CONVROT64_INVOKED: &str =
    "PyCapsule, PyCapsule, PyCapsule, int, bool, int, int, PyCapsule, float, int";
pub(crate) const QUANTIZE_INT8_CONVROT_STAGED_SIGNATURE: &str = "quantize_int8_convrot_staged(input: ndarray[shape=(*, *), device='cuda'], rotated: ndarray[shape=(*, *), device='cuda'], partial_absmax: ndarray[dtype=float32, shape=(*, *), device='cuda'], output: ndarray[dtype=int8, shape=(*, *), device='cuda'], scales: ndarray[dtype=float32, shape=(*, *), device='cuda'], group_size: int, stochastic: bool, seed: int, stream_ptr: int) -> None";
pub(crate) const QUANTIZE_INT8_CONVROT_STAGED_INVOKED: &str =
    "PyCapsule, PyCapsule, PyCapsule, PyCapsule, PyCapsule, int, bool, int, int";
pub(crate) const QUANTIZE_INT4_ROWWISE_SIGNATURE: &str = "quantize_int4_rowwise(input: ndarray[shape=(*, *), device='cuda'], output: ndarray[dtype=int8, shape=(*, *), device='cuda'], scales: ndarray[dtype=float32, shape=(*, *), device='cuda'], stochastic: bool, seed: int, stream_ptr: int) -> None";
pub(crate) const QUANTIZE_INT4_ROWWISE_INVOKED: &str =
    "PyCapsule, PyCapsule, PyCapsule, bool, int, int";
pub(crate) const QUANTIZE_INT4_ROWWISE_CONVROT64_SIGNATURE: &str = "quantize_int4_rowwise_convrot64(input: ndarray[shape=(*, *), device='cuda'], output: ndarray[dtype=int8, shape=(*, *), device='cuda'], scales: ndarray[dtype=float32, shape=(*, *), device='cuda'], group_size: int, stochastic: bool, seed: int, stream_ptr: int) -> None";
pub(crate) const QUANTIZE_INT4_ROWWISE_CONVROT64_INVOKED: &str =
    "PyCapsule, PyCapsule, PyCapsule, int, bool, int, int";
pub(crate) const QUANTIZE_W4A8_CONVROT_SIGNATURE: &str = "quantize_w4a8_convrot(rotated: ndarray[shape=(*, *), device='cuda'], codebook: ndarray[dtype=float32, shape=(*), device='cuda'], packed: ndarray[dtype=int8, shape=(*, *), device='cuda'], s_rel: ndarray[dtype=uint8, shape=(*, *), device='cuda'], s_channel: ndarray[dtype=float32, shape=(*), device='cuda'], stochastic: bool, seed: int, stream_ptr: int) -> None";
pub(crate) const QUANTIZE_W4A8_CONVROT_INVOKED: &str =
    "PyCapsule, PyCapsule, PyCapsule, PyCapsule, PyCapsule, bool, int, int";
pub(crate) const QUANTIZE_WXA8_CONVROT_FUSED_SIGNATURE: &str = "quantize_wxa8_convrot_fused(weight: ndarray[shape=(*, *), device='cuda'], codebook: ndarray[dtype=float32, device='cuda'], packed: ndarray[dtype=int8, shape=(*, *), device='cuda'], s_rel: ndarray[dtype=uint8, shape=(*, *), device='cuda'], s_channel: ndarray[dtype=float32, shape=(*), device='cuda'], bits: int, group_size: int, stochastic: bool, seed: int, stream_ptr: int) -> bool";
pub(crate) const QUANTIZE_WXA8_CONVROT_FUSED_INVOKED: &str =
    "PyCapsule, PyCapsule, PyCapsule, PyCapsule, PyCapsule, int, int, bool, int, int";
pub(crate) const QUANTIZE_NVFP4_SIGNATURE: &str = "quantize_nvfp4(input: ndarray[shape=(*, *), device='cuda'], global_scale: ndarray[device='cuda'], output: ndarray[device='cuda'], block_scales: ndarray[device='cuda'], epsilon: float, pad_16x: bool = False, hi_first: bool = True, stream_ptr: int) -> None";
pub(crate) const QUANTIZE_NVFP4_INVOKED: &str =
    "PyCapsule, PyCapsule, PyCapsule, PyCapsule, float, bool, bool, int";
pub(crate) const DEQUANTIZE_NVFP4_SIGNATURE: &str = "dequantize_nvfp4(input: ndarray[shape=(*, *), device='cuda'], global_scale: ndarray[device='cuda'], block_scales: ndarray[device='cuda'], output: ndarray[shape=(*, *), device='cuda'], output_dtype_code: int, hi_first: bool = True, stream_ptr: int) -> None";
pub(crate) const DEQUANTIZE_NVFP4_INVOKED: &str =
    "PyCapsule, PyCapsule, PyCapsule, PyCapsule, int, bool, int";
