#ifndef __CNN_BUILTIN_MATH_CU_H__
#define __CNN_BUILTIN_MATH_CU_H__

#define CNN_SCALAR_ACTIV_DEF(name)                          \
    void cnn_##name##_gpu(float* dst, float* src, int len); \
    void cnn_##name##_grad_gpu(float* dst, float* src, float* cache, int len);

#ifdef __cplusplus
extern "C"
{
#endif

    void cnn_max_gpu(float* maxPtr, float* vec, int len, float* buf);
    void cnn_sum_gpu(float* sumPtr, float* vec, int len, float* buf);

    void cnn_add_gpu(float* dst, float* src, int len, float addend);
    void cnn_exp_gpu(float* dst, float* src, int len);
    void cnn_div_gpu(float* dst, float* src, int len, float divider);
    void cnn_fmaxf_gpu(float* dst, float* src, int len, float num);
    void cnn_elemwise_product_gpu(float* dst, float* src1, float* src2,
                                  int len);

    CNN_SCALAR_ACTIV_DEF(relu);
    CNN_SCALAR_ACTIV_DEF(swish);
    CNN_SCALAR_ACTIV_DEF(sigmoid);
    CNN_SCALAR_ACTIV_DEF(tanh);
    CNN_SCALAR_ACTIV_DEF(gaussian);
    CNN_SCALAR_ACTIV_DEF(bent_identity);
    CNN_SCALAR_ACTIV_DEF(softplus);
    CNN_SCALAR_ACTIV_DEF(softsign);
    CNN_SCALAR_ACTIV_DEF(sinc);
    CNN_SCALAR_ACTIV_DEF(sin);

    void cnn_smax_grad_gpu(float* dst, float* cache, int len);

#ifdef __cplusplus
}
#endif

#endif
