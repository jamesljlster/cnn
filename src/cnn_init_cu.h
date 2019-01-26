#ifndef __CNN_INIT_CU_H__
#define __CNN_INIT_CU_H__

#include "cnn_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    __global__ void cnn_conv_unroll_2d_valid_cu(int* indexMap, int dstHeight,
                                                int dstWidth, int kSize,
                                                int srcHeight, int srcWidth,
                                                int srcCh);

    __global__ void cnn_conv_unroll_2d_same_cu(int* indexMap, int dstHeight,
                                               int dstWidth, int kSize,
                                               int srcHeight, int srcWidth,
                                               int srcCh);

    __global__ void cnn_bn_init_cu(float* bnVar, int ch, float rInit,
                                   float bInit);

#ifdef __cplusplus
}
#endif

#endif
