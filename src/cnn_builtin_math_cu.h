#ifndef __CNN_BUILTIN_MATH_CU_H__
#define __CNN_BUILTIN_MATH_CU_H__

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

    void cnn_relu_gpu(float* dst, float* src, int len);
    void cnn_swish_gpu(float* dst, float* src, int len);
    void cnn_sigmoid_gpu(float* dst, float* src, int len);
    void cnn_tanh_gpu(float* dst, float* src, int len);
    void cnn_gaussian_gpu(float* dst, float* src, int len);
    void cnn_bent_identity_gpu(float* dst, float* src, int len);
    void cnn_softplus_gpu(float* dst, float* src, int len);
    void cnn_softsign_gpu(float* dst, float* src, int len);
    void cnn_sinc_gpu(float* dst, float* src, int len);
    void cnn_sin_gpu(float* dst, float* src, int len);

    void cnn_smax_grad_gpu(float* dst, float* cache, int len);
    void cnn_relu_grad_gpu(float* dst, float* src, int len);
    void cnn_swish_grad_gpu(float* dst, float* src, float* cache, int len);
    void cnn_sigmoid_grad_gpu(float* dst, float* cache, int len);
    void cnn_tanh_grad_gpu(float* dst, float* cache, int len);
    void cnn_gaussian_grad_gpu(float* dst, float* src, float* cache, int len);
    void cnn_bent_identity_grad_gpu(float* dst, float* src, int len);
    void cnn_softplus_grad_gpu(float* dst, float* src, int len);
    void cnn_softsign_grad_gpu(float* dst, float* src, int len);
    void cnn_sinc_grad_gpu(float* dst, float* src, int len);
    void cnn_sin_grad_gpu(float* dst, float* src, int len);

#ifdef __cplusplus
}
#endif

#endif
