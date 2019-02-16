#ifndef __CNN_CALC_CU_H__
#define __CNN_CALC_CU_H__

#ifdef __cplusplus
extern "C"
{
#endif

    void cnn_drop_gpu(float* dst, float* src, int* mask, int size, float scale);
    void cnn_drop_grad_gpu(float* gradDst, float* gradSrc, int* mask, int size,
                           float scale);

    void cnn_map_gpu(float* dst, float* src, int* map, int len);
    void cnn_map_inv_gpu(float* dst, float* src, int* map, int len);

    void cnn_pool_2d_max_gpu(float* dst, int* indexMat, int dstWidth,
                             int dstHeight, int poolSize, float* src,
                             int srcWidth, int srcHeight, int channel);
    void cnn_pool_2d_max_grad_gpu(float* grad, int* indexMat, float* gradIn,
                                  int size);

#ifdef __cplusplus
}
#endif

#endif
