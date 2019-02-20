#ifndef __CNN_DROP_H__
#define __CNN_DROP_H__

#include <stdlib.h>
#include <string.h>

#include "cnn_types.h"

#ifdef CNN_WITH_CUDA
#include <cuda_runtime.h>

void cnn_drop_gpu(float* dst, float* src, int* mask, int size, float scale);
void cnn_drop_grad_gpu(float* gradDst, float* gradSrc, int* mask, int size,
                       float scale);
#endif

static inline void cnn_drop(float* dst, float* src, int* mask, int size,
                            float scale)
{
    for (int __i = 0; __i < size; __i++)
    {
        if (mask[__i] > 0)
        {
            dst[__i] = src[__i] * scale;
        }
        else
        {
            dst[__i] = 0;
        }
    }
}

static inline void cnn_drop_grad(float* gradDst, float* gradSrc, int* mask,
                                 int size, float scale)
{
    for (int __i = 0; __i < size; __i++)
    {
        if (mask[__i] > 0)
        {
            gradDst[__i] += gradSrc[__i] * scale;
        }
    }
}

static inline void cnn_forward_drop(union CNN_LAYER* layerRef,
                                    struct CNN_CONFIG* cfgRef, int layerIndex)
{
    int size = layerRef[layerIndex].outMat.data.rows *
               layerRef[layerIndex].outMat.data.cols;
    int* mask = layerRef[layerIndex].drop.mask;

#ifdef CNN_WITH_CUDA
    int* maskGpu = layerRef[layerIndex].drop.maskGpu;
#endif

    float rate = cfgRef->layerCfg[layerIndex].drop.rate;

    // Generate dropout mask
    for (int j = 0; j < size; j++)
    {
        if ((float)rand() / (float)RAND_MAX >= rate)
        {
            mask[j] = 1;
        }
        else
        {
            mask[j] = 0;
        }
    }

#ifdef CNN_WITH_CUDA
    cudaMemcpy(maskGpu, mask, size * sizeof(int), cudaMemcpyHostToDevice);
    cnn_drop_gpu(layerRef[layerIndex].outMat.data.mat,
                 layerRef[layerIndex - 1].outMat.data.mat, maskGpu, size,
                 cfgRef->layerCfg[layerIndex].drop.scale);
#else
    cnn_drop(layerRef[layerIndex].outMat.data.mat,
             layerRef[layerIndex - 1].outMat.data.mat, mask, size,
             cfgRef->layerCfg[layerIndex].drop.scale);
#endif
}

static inline void cnn_backward_drop(union CNN_LAYER* layerRef,
                                     struct CNN_CONFIG* cfgRef, int layerIndex)
{
    if (layerIndex > 1)
    {
        int size = layerRef[layerIndex].outMat.data.rows *
                   layerRef[layerIndex].outMat.data.cols;

        // Find layer gradient
#ifdef CNN_WITH_CUDA
        int* maskGpu = layerRef[layerIndex].drop.maskGpu;
        cudaMemset(layerRef[layerIndex - 1].outMat.data.grad, 0,
                   size * sizeof(float));
        cnn_drop_grad_gpu(layerRef[layerIndex - 1].outMat.data.grad,
                          layerRef[layerIndex].outMat.data.grad, maskGpu, size,
                          cfgRef->layerCfg[layerIndex].drop.scale);
#else
        int* mask = layerRef[layerIndex].drop.mask;
        memset(layerRef[layerIndex - 1].outMat.data.grad, 0,
               size * sizeof(float));
        cnn_drop_grad(layerRef[layerIndex - 1].outMat.data.grad,
                      layerRef[layerIndex].outMat.data.grad, mask, size,
                      cfgRef->layerCfg[layerIndex].drop.scale);
#endif
    }
}

#endif
