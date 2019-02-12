#ifndef __CNN_CALC_CU_H__
#define __CNN_CALC_CU_H__

#ifdef __cplusplus
extern "C"
{
#endif

    void cnn_map_gpu(float* dst, float* src, int* map, int len);
    void cnn_map_inv_gpu(float* dst, float* src, int* map, int len);

    void cnn_drop_gpu(float* dst, float* src, int* mask, int size, float scale);
    void cnn_drop_grad_gpu(float* gradDst, float* gradSrc, int* mask, int size,
                           float scale);

#ifdef __cplusplus
}
#endif

#endif
