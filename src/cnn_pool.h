#ifndef __CNN_POOL_H__
#define __CNN_POOL_H__

#include <string.h>

#include "cnn_types.h"

#ifdef CNN_WITH_CUDA
#include <cuda_runtime.h>

void cnn_pool_2d_max_gpu(float* dst, int* indexMat, int dstWidth, int dstHeight,
                         int poolSize, float* src, int srcWidth, int srcHeight,
                         int channel);
void cnn_pool_2d_max_grad_gpu(float* grad, int* indexMat, float* gradIn,
                              int size);
#endif

static inline void cnn_pool_2d_max(float* dst, int* indexMat, int dstHeight,
                                   int dstWidth, float* src, int srcWidth,
                                   int srcHeight, int poolSize, int channel)
{
    int __dstImSize = dstHeight * dstWidth;
    int __srcImSize = srcHeight * srcWidth;
    int __size = channel * __dstImSize;

#pragma omp parallel for
    for (int __i = 0; __i < __size; __i++)
    {
        int __ch = __i / __dstImSize;
        int __h = (__i % __dstImSize) / dstWidth;
        int __w = __i % dstWidth;

        int __dstChShift = __ch * __dstImSize;
        int __srcChShift = __ch * __srcImSize;

        float __tmp, __max;
        int __maxIndex, __index;

        __index = (__h * poolSize) * srcWidth + (__w * poolSize) + __srcChShift;
        __max = src[__index];
        __maxIndex = __index;
        for (int __poolH = 0; __poolH < poolSize; __poolH++)
        {
            for (int __poolW = 0; __poolW < poolSize; __poolW++)
            {
                __index = ((__h * poolSize) + __poolH) * srcWidth +
                          ((__w * poolSize) + __poolW) + __srcChShift;
                __tmp = src[__index];
                if (__tmp > __max)
                {
                    __max = __tmp;
                    __maxIndex = __index;
                }
            }
        }

        __index = __h * dstWidth + __w + __dstChShift;
        dst[__index] = __max;
        indexMat[__index] = __maxIndex;
    }
}

static inline void cnn_pool_2d_max_grad(float* grad, int* indexMat,
                                        float* iGrad, int iGradRows,
                                        int iGradCols, int iCh)
{
    int size = iGradRows * iGradCols * iCh;
#pragma omp parallel for
    for (int __i = 0; __i < size; __i++)
    {
        int __index = indexMat[__i];
        grad[__index] += iGrad[__i];
    }
}

static inline void cnn_forward_pool(union CNN_LAYER* layerRef,
                                    struct CNN_CONFIG* cfgRef, int layerIndex)
{
    // Clear outputs
#ifdef CNN_WITH_CUDA
    cudaMemset
#else
    memset
#endif
        (layerRef[layerIndex].outMat.data.mat, 0,
         sizeof(float) * layerRef[layerIndex].outMat.data.rows *
             layerRef[layerIndex].outMat.data.cols);

    for (int j = 0; j < cfgRef->batch; j++)
    {
        int srcShift = j * layerRef[layerIndex - 1].outMat.data.cols;
        int dstShift = j * layerRef[layerIndex].outMat.data.cols;

        float* srcPtr = layerRef[layerIndex - 1].outMat.data.mat + srcShift;
        float* dstPtr = layerRef[layerIndex].outMat.data.mat + dstShift;

#ifdef CNN_WITH_CUDA
        cnn_pool_2d_max_gpu(dstPtr,
                            layerRef[layerIndex].pool.indexMat + dstShift,
                            layerRef[layerIndex].outMat.width,
                            layerRef[layerIndex].outMat.height,
                            cfgRef->layerCfg[layerIndex].pool.size, srcPtr,
                            layerRef[layerIndex - 1].outMat.width,
                            layerRef[layerIndex - 1].outMat.height,
                            layerRef[layerIndex].outMat.channel);
#else
        cnn_pool_2d_max(dstPtr, &layerRef[layerIndex].pool.indexMat[dstShift],
                        layerRef[layerIndex].outMat.height,
                        layerRef[layerIndex].outMat.width, srcPtr,
                        layerRef[layerIndex - 1].outMat.height,
                        layerRef[layerIndex - 1].outMat.width,
                        cfgRef->layerCfg[layerIndex].pool.size,
                        layerRef[layerIndex].outMat.channel);
#endif
    }
}

static inline void cnn_backward_pool(union CNN_LAYER* layerRef,
                                     struct CNN_CONFIG* cfgRef, int layerIndex)
{
    int srcShift, dstShift;
    float* srcPtr;

    if (layerIndex > 1)
    {
        // Zero layer gradient
#ifdef CNN_WITH_CUDA
        cudaMemset
#else
        memset
#endif
            (layerRef[layerIndex - 1].outMat.data.grad, 0,
             sizeof(float) * layerRef[layerIndex - 1].outMat.data.rows *
                 layerRef[layerIndex - 1].outMat.data.cols);

        for (int j = 0; j < cfgRef->batch; j++)
        {
            srcShift = j * layerRef[layerIndex].outMat.data.cols;
            dstShift = j * layerRef[layerIndex - 1].outMat.data.cols;

            srcPtr = layerRef[layerIndex].outMat.data.grad + srcShift;

            // Find layer gradient
#ifdef CNN_WITH_CUDA
            cnn_pool_2d_max_grad_gpu(
                layerRef[layerIndex - 1].outMat.data.grad + dstShift,
                layerRef[layerIndex].pool.indexMat + srcShift, srcPtr,
                layerRef[layerIndex].outMat.width *
                    layerRef[layerIndex].outMat.height *
                    layerRef[layerIndex].outMat.channel);
#else
            cnn_pool_2d_max_grad(
                &layerRef[layerIndex - 1].outMat.data.grad[dstShift],
                &layerRef[layerIndex].pool.indexMat[srcShift], srcPtr,
                layerRef[layerIndex].outMat.height,
                layerRef[layerIndex].outMat.width,
                layerRef[layerIndex].outMat.channel);
#endif
        }
    }
}

#endif
