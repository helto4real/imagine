static __global__ void paired_rms_norm_rope_apply_bf16(
        const nv_bfloat16 * q, const nv_bfloat16 * k,
        const float * qw, const float * kw, const float * pe,
        nv_bfloat16 * qdst, nv_bfloat16 * kdst,
        int64_t heads, int64_t tokens, float qeps, float keps, RopeApplyStrides strides) {
    const int col = threadIdx.x;
    const int64_t row = blockIdx.x;
    const int64_t head = row % heads;
    const int64_t token = (row / heads) % tokens;
    const int64_t batch = row / (heads * tokens);
    const int64_t offset = head * strides.x[1] + token * strides.x[2] + batch * strides.x[3];
    const nv_bfloat16 * qi = q + offset, * ki = k + offset;
    ggml_cuda_pdl_sync();
    const float qv = qi[col], kv = ki[col];
    float qs = 0.0f, ks = 0.0f;
    qs += qv * qv;
    ks += kv * kv;
    extern __shared__ float partial[];
    // Preserve each 128-thread reduction; disjoint scratch avoids reuse races.
    qs = block_reduce<block_reduce_method::SUM, 128>(qs, partial);
    ks = block_reduce<block_reduce_method::SUM, 128>(ks, partial + 32);
    const float qscale = rsqrtf(qs / 128.0f + qeps);
    const float kscale = rsqrtf(ks / 128.0f + keps);
    if (col < 64) {
        const int i = 2 * col;
        const float qa = __fmul_rn(__fmul_rn(qscale, qi[i]), qw[i]);
        const float qb = __fmul_rn(__fmul_rn(qscale, qi[i + 1]), qw[i + 1]);
        const float ka = __fmul_rn(__fmul_rn(kscale, ki[i]), kw[i]);
        const float kb = __fmul_rn(__fmul_rn(kscale, ki[i + 1]), kw[i + 1]);
        const float * matrix = pe + col * strides.pe[2] + token * strides.pe[3];
        const float m00 = matrix[0], m01 = matrix[strides.pe[0]];
        const float m10 = matrix[strides.pe[1]], m11 = matrix[strides.pe[0] + strides.pe[1]];
        const int64_t out = ((batch * heads + head) * tokens + token) * 128 + i;
        qdst[out] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(qa, m00), __fmul_rn(qb, m01)));
        qdst[out + 1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(qa, m10), __fmul_rn(qb, m11)));
        kdst[out] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(ka, m00), __fmul_rn(kb, m01)));
        kdst[out + 1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(ka, m10), __fmul_rn(kb, m11)));
    }
    ggml_cuda_pdl_lc();
}
