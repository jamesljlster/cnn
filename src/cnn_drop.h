#ifndef __CNN_DROP_H__
#define __CNN_DROP_H__

#include <stdlib.h>
#include <string.h>

#include "cnn_init.h"
#include "cnn_macro.h"
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

static inline void cnn_recall_drop(union CNN_LAYER* layerRef,
                                   struct CNN_CONFIG* cfgRef, int layerIndex)
{
    struct CNN_MAT* outData = &layerRef[layerIndex].outMat.data;
    struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

    int size = outData->rows * outData->cols * sizeof(float);

#ifdef CNN_WITH_CUDA
    cudaMemcpy(outData->mat, preOutData->mat, size, cudaMemcpyDeviceToDevice);
#else
    memcpy(outData->mat, preOutData->mat, size);
#endif
}

static inline void cnn_forward_drop(union CNN_LAYER* layerRef,
                                    struct CNN_CONFIG* cfgRef, int layerIndex)
{
#ifdef CNN_WITH_CUDA
    struct CNN_LAYER_DROP* layerPtr = &layerRef[layerIndex].drop;

    struct CNN_MAT* outData = &layerPtr->outMat.data;
    struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

    cnn_assert_cudnn(cudnnDropoutForward(         //
        cnnInit.cudnnHandle, layerPtr->dropDesc,  //
        layerPtr->ten, preOutData->mat,           //
        layerPtr->ten, outData->mat,              //
        layerPtr->rsvSpace, layerPtr->rsvSize));
#else
    int size = layerRef[layerIndex].outMat.data.rows *
               layerRef[layerIndex].outMat.data.cols;
    int* mask = layerRef[layerIndex].drop.mask;

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

    cnn_drop(layerRef[layerIndex].outMat.data.mat,
             layerRef[layerIndex - 1].outMat.data.mat, mask, size,
             cfgRef->layerCfg[layerIndex].drop.scale);

#endif
}

static inline void cnn_backward_drop(union CNN_LAYER* layerRef,
                                     struct CNN_CONFIG* cfgRef, int layerIndex)
{
    // Find layer gradient
    if (layerIndex > 1)
    {
#ifdef CNN_WITH_CUDA
        struct CNN_LAYER_DROP* layerPtr = &layerRef[layerIndex].drop;

        struct CNN_MAT* outData = &layerPtr->outMat.data;
        struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

        cnn_assert_cudnn(cudnnDropoutBackward(        //
            cnnInit.cudnnHandle, layerPtr->dropDesc,  //
            layerPtr->ten, outData->grad,             //
            layerPtr->ten, preOutData->grad,          //
            layerPtr->rsvSpace, layerPtr->rsvSize));

#else
        int size = layerRef[layerIndex].outMat.data.rows *
                   layerRef[layerIndex].outMat.data.cols;

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
