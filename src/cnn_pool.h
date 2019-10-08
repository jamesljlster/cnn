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

static inline void cnn_pool_2d_avg(           //
    float* dst, int dstHeight, int dstWidth,  //
    float* src, int srcHeight, int srcWidth,  //
    int batch, int channel, int poolSize)
{
    for (int i = 0; i < batch * channel * dstHeight * dstWidth; i++)
    {
        int w = i % dstWidth;
        int h = (i / dstWidth) % dstHeight;
        int c = (i / (dstHeight * dstWidth)) % channel;
        int n = i / (channel * dstHeight * dstWidth);

        float sum = 0;

        for (int poolH = 0; poolH < poolSize; poolH++)
        {
            for (int poolW = 0; poolW < poolSize; poolW++)
            {
                sum += src[n * (channel * srcHeight * srcWidth) +  //
                           c * (srcHeight * srcWidth) +            //
                           (h * poolSize + poolH) * srcWidth +     //
                           (w * poolSize + poolW)];
            }
        }

        dst[n * (channel * dstHeight * dstWidth) +  //
            c * (dstHeight * dstWidth) +            //
            h * dstWidth +                          //
            w] = sum / (poolSize * poolSize);
    }
}

static inline void cnn_pool_2d_avg_grad(                  //
    float* gradOut, int gradOutHeight, int gradOutWidth,  //
    float* gradIn, int gradInHeight, int gradInWidth,     //
    int batch, int channel, int poolSize)
{
    for (int i = 0; i < batch * channel * gradInHeight * gradInWidth; i++)
    {
        int w = i % gradInWidth;
        int h = (i / gradInWidth) % gradInHeight;
        int c = (i / (gradInHeight * gradInWidth)) % channel;
        int n = i / (channel * gradInHeight * gradInWidth);

        float gradTmp = gradIn[n * (channel * gradInHeight * gradInWidth) +  //
                               c * (gradInHeight * gradInWidth) +            //
                               h * gradInWidth +                             //
                               w] /
                        (poolSize * poolSize);

        for (int poolH = 0; poolH < poolSize; poolH++)
        {
            for (int poolW = 0; poolW < poolSize; poolW++)
            {
                gradOut[n * (channel * gradOutHeight * gradOutWidth) +  //
                        c * (gradOutHeight * gradOutWidth) +            //
                        (h * poolSize + poolH) * gradOutWidth +         //
                        (w * poolSize + poolW)] = gradTmp;
            }
        }
    }
}

static inline void cnn_pool_2d_max(float* dst, int* indexMat, int dstHeight,
                                   int dstWidth, float* src, int srcWidth,
                                   int srcHeight, int poolSize, int channel)
{
    int __dstImSize = dstHeight * dstWidth;
    int __srcImSize = srcHeight * srcWidth;

    for (int __ch = 0; __ch < channel; __ch++)
    {
        int __dstChShift = __ch * __dstImSize;
        int __srcChShift = __ch * __srcImSize;

        for (int __h = 0; __h < dstHeight; __h++)
        {
            for (int __w = 0; __w < dstWidth; __w++)
            {
                float __tmp, __max;
                int __maxIndex, __index;

                __index = (__h * poolSize) * srcWidth + (__w * poolSize) +
                          __srcChShift;
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
    }
}

static inline void cnn_pool_2d_max_grad(float* grad, int* indexMat,
                                        float* iGrad, int iGradRows,
                                        int iGradCols, int iCh)
{
    int size = iGradRows * iGradCols * iCh;
    for (int __i = 0; __i < size; __i++)
    {
        grad[indexMat[__i]] += iGrad[__i];
    }
}

static inline void cnn_forward_pool(union CNN_LAYER* layerRef,
                                    struct CNN_CONFIG* cfgRef, int layerIndex)
{
#ifdef CNN_WITH_CUDA
    float alpha = 1.0;
    float beta = 0.0;

    struct CNN_LAYER_POOL* layerPtr = &layerRef[layerIndex].pool;

    struct CNN_MAT* outData = &layerPtr->outMat.data;
    struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

    cnn_assert_cudnn(cudnnPoolingForward(         //
        cnnInit.cudnnHandle, layerPtr->poolDesc,  //
        &alpha,                                   //
        layerPtr->srcTen, preOutData->mat,        //
        &beta,                                    //
        layerPtr->dstTen, outData->mat));
#else
    if (cfgRef->layerCfg[layerIndex].pool.poolType == CNN_POOL_MAX)
    {
        // Clear outputs
        memset(layerRef[layerIndex].outMat.data.mat, 0,
               sizeof(float) * layerRef[layerIndex].outMat.data.rows *
                   layerRef[layerIndex].outMat.data.cols);

        for (int j = 0; j < cfgRef->batch; j++)
        {
            int srcShift = j * layerRef[layerIndex - 1].outMat.data.cols;
            int dstShift = j * layerRef[layerIndex].outMat.data.cols;

            float* srcPtr = layerRef[layerIndex - 1].outMat.data.mat + srcShift;
            float* dstPtr = layerRef[layerIndex].outMat.data.mat + dstShift;

            cnn_pool_2d_max(dstPtr,
                            &layerRef[layerIndex].pool.indexMat[dstShift],
                            layerRef[layerIndex].outMat.height,
                            layerRef[layerIndex].outMat.width, srcPtr,
                            layerRef[layerIndex - 1].outMat.height,
                            layerRef[layerIndex - 1].outMat.width,
                            cfgRef->layerCfg[layerIndex].pool.size,
                            layerRef[layerIndex].outMat.channel);
        }
    }
    else
    {
        cnn_pool_2d_avg(layerRef[layerIndex].outMat.data.mat,
                        layerRef[layerIndex].outMat.height,
                        layerRef[layerIndex].outMat.width,
                        layerRef[layerIndex - 1].outMat.data.mat,
                        layerRef[layerIndex - 1].outMat.height,
                        layerRef[layerIndex - 1].outMat.width, cfgRef->batch,
                        layerRef[layerIndex].outMat.channel,
                        cfgRef->layerCfg[layerIndex].pool.size);
    }
#endif
}

static inline void cnn_backward_pool(union CNN_LAYER* layerRef,
                                     struct CNN_CONFIG* cfgRef, int layerIndex)
{
#ifdef CNN_WITH_CUDA
    float alpha = 1.0;
    float beta = 0.0;

    struct CNN_LAYER_POOL* layerPtr = &layerRef[layerIndex].pool;

    struct CNN_MAT* outData = &layerPtr->outMat.data;
    struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;
#else
    int srcShift, dstShift;
    float* srcPtr;
#endif

    if (layerIndex > 1)
    {
#ifdef CNN_WITH_CUDA
        cnn_assert_cudnn(cudnnPoolingBackward(        //
            cnnInit.cudnnHandle, layerPtr->poolDesc,  //
            &alpha,                                   //
            layerPtr->dstTen, outData->mat,           //
            layerPtr->dstTen, outData->grad,          //
            layerPtr->srcTen, preOutData->mat,        //
            &beta,                                    //
            layerPtr->srcTen, preOutData->grad));
#else
        if (cfgRef->layerCfg[layerIndex].pool.poolType == CNN_POOL_MAX)
        {
            // Zero layer gradient
            memset(layerRef[layerIndex - 1].outMat.data.grad, 0,
                   sizeof(float) * layerRef[layerIndex - 1].outMat.data.rows *
                       layerRef[layerIndex - 1].outMat.data.cols);

            for (int j = 0; j < cfgRef->batch; j++)
            {
                srcShift = j * layerRef[layerIndex].outMat.data.cols;
                dstShift = j * layerRef[layerIndex - 1].outMat.data.cols;

                srcPtr = layerRef[layerIndex].outMat.data.grad + srcShift;

                // Find layer gradient
                cnn_pool_2d_max_grad(
                    &layerRef[layerIndex - 1].outMat.data.grad[dstShift],
                    &layerRef[layerIndex].pool.indexMat[srcShift], srcPtr,
                    layerRef[layerIndex].outMat.height,
                    layerRef[layerIndex].outMat.width,
                    layerRef[layerIndex].outMat.channel);
            }
        }
        else
        {
            cnn_pool_2d_avg_grad(layerRef[layerIndex - 1].outMat.data.grad,
                                 layerRef[layerIndex - 1].outMat.height,
                                 layerRef[layerIndex - 1].outMat.width,
                                 layerRef[layerIndex].outMat.data.grad,
                                 layerRef[layerIndex].outMat.height,
                                 layerRef[layerIndex].outMat.width,
                                 cfgRef->batch,
                                 layerRef[layerIndex].outMat.channel,
                                 cfgRef->layerCfg[layerIndex].pool.size);
        }
#endif
    }
}

#endif
