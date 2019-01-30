#ifndef __CNN_CALC_CU_H__
#define __CNN_CALC_CU_H__

#ifdef __cplusplus
extern "C"
{
#endif

    void cnn_max_gpu(float* maxPtr, float* vec, int len, float* buf);
    void cnn_sum_gpu(float* sumPtr, float* vec, int len, float* buf);

    void cnn_add_gpu(float* src, float* dst, int len, float addend);
    void cnn_exp_gpu(float* src, float* dst, int len);
    void cnn_div_gpu(float* src, float* dst, int len, float divider);

    void cnn_smax_grad_gpu(float* dst, float* cache, int len);

#ifdef __cplusplus
}
#endif

#endif
