#ifndef __CNN_CONV_H__
#define __CNN_CONV_H__

#include <cblas.h>
#include <string.h>

#include "cnn_macro.h"
#include "cnn_types.h"

#ifdef CNN_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cnn_init.h"

void cnn_map_gpu(float* dst, float* src, int* map, int len);
void cnn_map_inv_gpu(float* dst, float* src, int* map, int len);
#endif

static inline void cnn_conv_unroll_2d_valid(int* indexMap, int dstHeight,
                                            int dstWidth, int kSize,
                                            int srcHeight, int srcWidth,
                                            int srcCh)
{
    int __kMemSize = kSize * kSize;
    int __srcImSize = srcHeight * srcWidth;
    int __indexMapCols = __kMemSize * srcCh;

    for (int __h = 0; __h < dstHeight; __h++)
    {
        int __dstRowShift = __h * dstWidth;

        for (int __w = 0; __w < dstWidth; __w++)
        {
            int __indexMapRow = __dstRowShift + __w;
            int __indexMemBase = __indexMapRow * __indexMapCols;

            for (int __ch = 0; __ch < srcCh; __ch++)
            {
                int __indexMemShiftBase = __indexMemBase + __kMemSize * __ch;
                int __srcChShift = __ch * __srcImSize;

                for (int __convH = 0; __convH < kSize; __convH++)
                {
                    int __indexMemShift = __indexMemShiftBase + __convH * kSize;
                    int __srcShift = (__h + __convH) * srcWidth + __srcChShift;

                    for (int __convW = 0; __convW < kSize; __convW++)
                    {
                        indexMap[__indexMemShift + __convW] =
                            __srcShift + (__w + __convW);
                    }
                }
            }
        }
    }
}

static inline void cnn_conv_unroll_2d_same(int* indexMap, int dstHeight,
                                           int dstWidth, int kSize,
                                           int srcHeight, int srcWidth,
                                           int srcCh)
{
    int __kMemSize = kSize * kSize;
    int __srcImSize = srcHeight * srcWidth;
    int __indexMapCols = __kMemSize * srcCh;

    int __convHBase = -kSize / 2;
    int __convWBase = -kSize / 2;

    for (int __h = 0; __h < dstHeight; __h++)
    {
        int __dstRowShift = __h * dstWidth;

        for (int __w = 0; __w < dstWidth; __w++)
        {
            int __indexMapRow = __dstRowShift + __w;
            int __indexMemBase = __indexMapRow * __indexMapCols;

            for (int __ch = 0; __ch < srcCh; __ch++)
            {
                int __indexMemShiftBase = __indexMemBase + __kMemSize * __ch;
                int __srcChShift = __ch * __srcImSize;

                for (int __convH = 0; __convH < kSize; __convH++)
                {
                    int __indexMemShift = __indexMemShiftBase + __convH * kSize;
                    int __convHIndex = __h + __convH + __convHBase;

                    if (__convHIndex >= 0 && __convHIndex < srcHeight)
                    {
                        int __srcShift = __convHIndex * srcWidth + __srcChShift;

                        for (int __convW = 0; __convW < kSize; __convW++)
                        {
                            int __convWIndex = __w + __convW + __convWBase;
                            if (__convWIndex >= 0 && __convWIndex < srcWidth)
                            {
                                int __tmpIndex = __srcShift + __convWIndex;

                                indexMap[__indexMemShift + __convW] =
                                    __tmpIndex;
                            }
                        }
                    }
                }
            }
        }
    }
}

static inline void cnn_conv_2d(float* dst, int dstHeight, int dstWidth,
                               float* kernel, int kSize, int chIn, int chOut,
                               float* src, int srcHeight, int srcWidth)
{
    int __kMemSize = kSize * kSize;
    int __filterSize = chIn * __kMemSize;
    int __dstImSize = dstHeight * dstWidth;
    int __srcImSize = srcHeight * srcWidth;

    for (int __chOut = 0; __chOut < chOut; __chOut++)
    {
        int __filterShift = __chOut * __filterSize;
        int __dstChShift = __chOut * __dstImSize;

        for (int __chIn = 0; __chIn < chIn; __chIn++)
        {
            int __kShiftBase = __chIn * __kMemSize + __filterShift;
            int __srcChShift = __chIn * __srcImSize;

            for (int __h = 0; __h < dstHeight; __h++)
            {
                int __dstShift = __h * dstWidth + __dstChShift;
                for (int __w = 0; __w < dstWidth; __w++)
                {
                    float __conv = 0;
                    for (int __convH = 0; __convH < kSize; __convH++)
                    {
                        int __kShift = __convH * kSize + __kShiftBase;
                        int __srcShift =
                            (__h + __convH) * srcWidth + __srcChShift;

                        for (int __convW = 0; __convW < kSize; __convW++)
                        {
                            __conv += kernel[__kShift + __convW] *
                                      src[__srcShift + (__w + __convW)];
                        }
                    }

                    dst[__dstShift + __w] += __conv;
                }
            }
        }
    }
}

static inline void cnn_conv_2d_grad(float* srcGrad, int srcHeight, int srcWidth,
                                    float* kernel, int kSize, int srcCh,
                                    int lCh, float* lGrad, int lHeight,
                                    int lWidth)
{
    int __kMemSize = kSize * kSize;
    int __filterSize = srcCh * __kMemSize;
    int __lImSize = lHeight * lWidth;
    int __srcImSize = srcHeight * srcWidth;

    for (int __lCh = 0; __lCh < lCh; __lCh++)
    {
        int __filterShift = __lCh * __filterSize;
        int __lChShift = __lCh * __lImSize;

        for (int __srcCh = 0; __srcCh < srcCh; __srcCh++)
        {
            int __srcChShift = __srcCh * __srcImSize;
            int __kShiftBase = __srcCh * __kMemSize + __filterShift;

            for (int __h = 0; __h < lHeight; __h++)
            {
                int __lShift = __h * lHeight + __lChShift;
                for (int __w = 0; __w < lWidth; __w++)
                {
                    for (int __convH = 0; __convH < kSize; __convH++)
                    {
                        int __kShift = __convH * kSize + __kShiftBase;
                        int __srcShift =
                            (__h + __convH) * srcWidth + __srcChShift;

                        for (int __convW = 0; __convW < kSize; __convW++)
                        {
                            srcGrad[__srcShift + (__w + __convW)] +=
                                lGrad[__lShift + __w] *
                                kernel[__kShift + __convW];
                        }
                    }
                }
            }
        }
    }
}

static inline void cnn_conv_2d_kernel_grad(float* lGrad, int lHeight,
                                           int lWidth, float* kGrad, int kSize,
                                           int lCh, int srcCh, float* src,
                                           int srcHeight, int srcWidth)
{
    int __kMemSize = kSize * kSize;
    int __filterSize = srcCh * __kMemSize;
    int __lImSize = lHeight * lWidth;
    int __srcImSize = srcHeight * srcWidth;

    for (int __lCh = 0; __lCh < lCh; __lCh++)
    {
        int __filterShift = __lCh * __filterSize;
        int __lChShift = __lCh * __lImSize;

        for (int __srcCh = 0; __srcCh < srcCh; __srcCh++)
        {
            int __srcChShift = __srcCh * __srcImSize;
            int __kShiftBase = __srcCh * __kMemSize + __filterShift;

            for (int __h = 0; __h < lHeight; __h++)
            {
                int __lShift = __h * lWidth + __lChShift;
                for (int __w = 0; __w < lWidth; __w++)
                {
                    for (int __convH = 0; __convH < kSize; __convH++)
                    {
                        int __kShift = __convH * kSize + __kShiftBase;
                        int __srcShift =
                            (__h + __convH) * srcWidth + __srcChShift;

                        for (int __convW = 0; __convW < kSize; __convW++)
                        {
                            kGrad[__kShift + __convW] +=
                                lGrad[__lShift + __w] *
                                src[__srcShift + (__w + __convW)];
                        }
                    }
                }
            }
        }
    }
}

static inline void cnn_forward_conv(union CNN_LAYER* layerRef,
                                    struct CNN_CONFIG* cfgRef, int layerIndex)
{
#ifdef CNN_WITH_CUDA
    float alpha = 1.0;
    float beta = 0.0;

    struct CNN_LAYER_CONV* layerPtr = &layerRef[layerIndex].conv;

    cnn_assert_cudnn(cudnnConvolutionForward(
        cnnInit.cudnnHandle,                                         //
        &alpha,                                                      //
        layerPtr->srcTen, layerRef[layerIndex - 1].outMat.data.mat,  //
        layerPtr->kernelTen, layerPtr->kernel.mat,                   //
        layerPtr->convDesc, layerPtr->convAlgoFW, cnnInit.wsData,
        cnnInit.wsSize,  //
        &beta,           //
        layerPtr->dstTen, layerPtr->outMat.data.mat));

#if defined(CNN_CONV_BIAS_FILTER)
    beta = 1.0;
    cnn_assert_cudnn(cudnnAddTensor(cnnInit.cudnnHandle,                    //
                                    &alpha,                                 //
                                    layerPtr->biasTen, layerPtr->bias.mat,  //
                                    &beta,                                  //
                                    layerPtr->dstTen,
                                    layerPtr->outMat.data.mat));
#endif

#else
    // Cache
    int mapRows =
        layerRef[layerIndex].outMat.width * layerRef[layerIndex].outMat.height;
    int mapCols = layerRef[layerIndex - 1].outMat.channel *
                  cfgRef->layerCfg[layerIndex].conv.size *
                  cfgRef->layerCfg[layerIndex].conv.size;
    int mapSize = mapRows * mapCols;
    int* indexMap = layerRef[layerIndex].conv.indexMap;

    int chOut = layerRef[layerIndex].outMat.channel;
    float* kernel = layerRef[layerIndex].conv.kernel.mat;

    // Clear outputs
    // memset(layerRef[layerIndex].outMat.data.mat, 0,
    //       sizeof(float) * layerRef[layerIndex].outMat.data.rows *
    //           layerRef[layerIndex].outMat.data.cols);

    for (int j = 0; j < cfgRef->batch; j++)
    {
        int srcShift = j * layerRef[layerIndex - 1].outMat.data.cols;
        int dstShift = j * layerRef[layerIndex].outMat.data.cols;
        int mapShift = j * mapSize;

        float* srcPtr = layerRef[layerIndex - 1].outMat.data.mat + srcShift;
        float* dstPtr = layerRef[layerIndex].outMat.data.mat + dstShift;
        float* mapPtr = layerRef[layerIndex].conv.unroll.mat + mapShift;

        for (int k = 0; k < mapSize; k++)
        {
            int tmpIndex = indexMap[k];
            if (tmpIndex >= 0)
            {
                mapPtr[k] = srcPtr[tmpIndex];
            }
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, chOut, mapRows,
                    mapCols, 1.0, kernel, mapCols, mapPtr, mapCols, 0.0, dstPtr,
                    mapRows);

        // Add bias
#if defined(CNN_CONV_BIAS_FILTER)
#ifdef DEBUG
#pragma message("cnn_forward_conv(): Enable convolution filter bias")
#endif
        for (int ch = 0; ch < chOut; ch++)
        {
            cblas_saxpy(
                mapRows, 1.0, &layerRef[layerIndex].conv.bias.mat[ch], 0,
                &layerRef[layerIndex].outMat.data.mat[dstShift + ch * mapRows],
                1);
        }
#elif defined(CNN_CONV_BIAS_LAYER)
#ifdef DEBUG
#pragma message("cnn_forward_conv(): Enable convolution layer bias")
#endif
        cblas_saxpy(layerRef[layerIndex].conv.bias.cols, 1.0,
                    layerRef[layerIndex].conv.bias.mat, 1,
                    &layerRef[layerIndex].outMat.data.mat[dstShift], 1);
#endif
    }
#endif
}

static inline void cnn_backward_conv(union CNN_LAYER* layerRef,
                                     struct CNN_CONFIG* cfgRef, int layerIndex)
{
#ifdef CNN_WITH_CUDA
    float alpha = 1.0;
    float beta = 1.0;

    struct CNN_LAYER_CONV* layerPtr = &layerRef[layerIndex].conv;

    cnn_assert_cudnn(cudnnConvolutionBackwardFilter(
        cnnInit.cudnnHandle,                                         //
        &alpha,                                                      //
        layerPtr->srcTen, layerRef[layerIndex - 1].outMat.data.mat,  //
        layerPtr->dstTen, layerRef[layerIndex].outMat.data.grad,     //
        layerPtr->convDesc, layerPtr->convAlgoBWFilter, cnnInit.wsData,
        cnnInit.wsSize,  //
        &beta,           //
        layerPtr->kernelTen, layerPtr->kernel.grad));

    cnn_assert_cudnn(cudnnConvolutionBackwardBias(
        cnnInit.cudnnHandle,                           //
        &alpha,                                        //
        layerPtr->dstTen, layerPtr->outMat.data.grad,  //
        &beta,                                         //
        layerPtr->biasTen, layerPtr->bias.grad));

#else
    // Cache
    int mapRows =
        layerRef[layerIndex].outMat.width * layerRef[layerIndex].outMat.height;
    int mapCols = layerRef[layerIndex - 1].outMat.channel *
                  cfgRef->layerCfg[layerIndex].conv.size *
                  cfgRef->layerCfg[layerIndex].conv.size;
    int mapSize = mapRows * mapCols;

    int* indexMap = layerRef[layerIndex].conv.indexMap;

    int chOut = layerRef[layerIndex].outMat.channel;
    float* kernel = layerRef[layerIndex].conv.kernel.mat;
    float* kGrad = layerRef[layerIndex].conv.kernel.grad;

    // Sum gradient
    for (int j = 0; j < cfgRef->batch; j++)
    {
        int gradShift = j * layerRef[layerIndex].outMat.data.cols;
        int mapShift = j * mapSize;

        float* gradPtr = &layerRef[layerIndex].outMat.data.grad[gradShift];
        float* mapPtr = &layerRef[layerIndex].conv.unroll.mat[mapShift];

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, chOut, mapCols,
                    mapRows, 1.0, gradPtr, mapRows, mapPtr, mapCols, 1.0, kGrad,
                    mapCols);

        // Sum bias gradient matrix
#if defined(CNN_CONV_BIAS_FILTER)
#ifdef DEBUG
#pragma message("cnn_forward_conv(): Enable convolution filter bias")
#endif
        for (int ch = 0; ch < chOut; ch++)
        {
            cblas_saxpy(mapRows, 1.0, &gradPtr[ch * mapRows], 1,
                        &layerRef[layerIndex].conv.bias.grad[ch], 0);
        }
#elif defined(CNN_CONV_BIAS_LAYER)
#ifdef DEBUG
#pragma message("cnn_forward_conv(): Enable convolution layer bias")
#endif
        cblas_saxpy(layerRef[layerIndex].conv.bias.cols, 1.0, gradPtr, 1,
                    layerRef[layerIndex].conv.bias.grad, 1);
#endif
    }
#endif

    // Find layer gradient
    if (layerIndex > 1)
    {
#ifdef CNN_WITH_CUDA
        cudaMemset(layerRef[layerIndex - 1].outMat.data.grad, 0,
                   sizeof(float) * layerRef[layerIndex - 1].outMat.data.rows *
                       layerRef[layerIndex - 1].outMat.data.cols);
        beta = 0.0;
        cnn_assert_cudnn(cudnnConvolutionBackwardData(
            cnnInit.cudnnHandle,                           //
            &alpha,                                        //
            layerPtr->kernelTen, layerPtr->kernel.mat,     //
            layerPtr->dstTen, layerPtr->outMat.data.grad,  //
            layerPtr->convDesc, layerPtr->convAlgoBWGrad, cnnInit.wsData,
            cnnInit.wsSize,  //
            &beta,           //
            layerPtr->srcTen, layerRef[layerIndex - 1].outMat.data.grad));
#else
        memset(layerRef[layerIndex - 1].outMat.data.grad, 0,
               sizeof(float) * layerRef[layerIndex - 1].outMat.data.rows *
                   layerRef[layerIndex - 1].outMat.data.cols);
        for (int j = 0; j < cfgRef->batch; j++)
        {
            int gradShift = j * layerRef[layerIndex].outMat.data.cols;
            int preGradShift = j * layerRef[layerIndex - 1].outMat.data.cols;
            int mapShift = j * mapSize;

            float* gradPtr = layerRef[layerIndex].outMat.data.grad + gradShift;
            float* preGradPtr =
                layerRef[layerIndex - 1].outMat.data.grad + preGradShift;
            float* mapPtr = layerRef[layerIndex].conv.unroll.grad + mapShift;

            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, mapRows,
                        mapCols, chOut, 1.0, gradPtr, mapRows, kernel, mapCols,
                        0.0, mapPtr, mapCols);

            for (int i = 0; i < mapSize; i++)
            {
                int tmpIndex = indexMap[i];
                if (tmpIndex >= 0)
                {
                    preGradPtr[tmpIndex] += mapPtr[i];
                }
            }
        }
#endif
    }
}

#endif
