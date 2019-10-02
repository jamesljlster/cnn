#ifndef __CNN_BN_H__
#define __CNN_BN_H__

#include <math.h>

#include "cnn_macro.h"
#include "cnn_types.h"

#ifdef CNN_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cnn_builtin_math_cu.h"
#include "cnn_init.h"
#endif

#define CNN_BN_EPS 1e-4

static inline void cnn_recall_bn(union CNN_LAYER* layerRef,
                                 struct CNN_CONFIG* cfgRef, int layerIndex)
{
    struct CNN_LAYER_BN* layerPtr = &layerRef[layerIndex].bn;

    struct CNN_MAT* outData = &layerPtr->outMat.data;
    struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

#ifdef CNN_WITH_CUDA
    float alpha = 1.0;
    float beta = 0.0;

    cnn_assert_cudnn(cudnnBatchNormalizationForwardInference(
        cnnInit.cudnnHandle, CUDNN_BATCHNORM_SPATIAL,  //
        &alpha, &beta,                                 //
        layerPtr->srcTen, preOutData->mat,             //
        layerPtr->srcTen, outData->mat,                //
        layerPtr->bnTen,                               //
        layerPtr->bnScale.mat, layerPtr->bnBias.mat,   //
        layerPtr->runMean.mat, layerPtr->runVar.mat,   //
        CNN_BN_EPS));

#else
    // Cache
    int channels = layerPtr->outMat.channel;
    int chSize = layerPtr->outMat.width * layerPtr->outMat.height;
    int batch = cfgRef->batch;

    // Batch normalization forward
    for (int ch = 0; ch < channels; ch++)
    {
        float mean = layerPtr->runMean.mat[ch];
        float var = layerPtr->runVar.mat[ch];

        float gamma = layerPtr->bnScale.mat[ch];
        float beta = layerPtr->bnBias.mat[ch];

        float stddev = sqrt(var + CNN_BN_EPS);

        // Process batch normalization
        for (int b = 0; b < batch; b++)
        {
            float* preImPtr =
                preOutData->mat + b * preOutData->cols + ch * chSize;
            float* imPtr = outData->mat + b * outData->cols + ch * chSize;

            for (int e = 0; e < chSize; e++)
            {
                float src = preImPtr[e];
                float out = (src - mean) / stddev;
                imPtr[e] = out * gamma + beta;
            }
        }
    }
#endif
}

static inline void cnn_forward_bn(union CNN_LAYER* layerRef,
                                  struct CNN_CONFIG* cfgRef, int layerIndex)
{
    struct CNN_LAYER_BN* layerPtr = &layerRef[layerIndex].bn;

    struct CNN_MAT* outData = &layerPtr->outMat.data;
    struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

#ifdef CNN_WITH_CUDA
    float alpha = 1.0;
    float beta = 0.0;

    cnn_assert_cudnn(cudnnBatchNormalizationForwardTraining(
        cnnInit.cudnnHandle, CUDNN_BATCHNORM_SPATIAL,                  //
        &alpha, &beta,                                                 //
        layerPtr->srcTen, preOutData->mat,                             //
        layerPtr->srcTen, outData->mat,                                //
        layerPtr->bnTen, layerPtr->bnScale.mat, layerPtr->bnBias.mat,  //
        cfgRef->layerCfg[layerIndex].bn.expAvgFactor,                  //
        layerPtr->runMean.mat, layerPtr->runVar.mat,                   //
        CNN_BN_EPS,                                                    //
        layerPtr->saveMean.mat, layerPtr->saveVar.mat));

#else
    // Cache
    int channels = layerPtr->outMat.channel;
    int chSize = layerPtr->outMat.width * layerPtr->outMat.height;
    int batch = cfgRef->batch;

    float sampleSize = batch * chSize;

    float expAvgFactor = cfgRef->layerCfg[layerIndex].bn.expAvgFactor;

    // Batch normalization forward
    for (int ch = 0; ch < channels; ch++)
    {
        float mean = 0;
        float var = 0;

        float gamma = layerPtr->bnScale.mat[ch];
        float beta = layerPtr->bnBias.mat[ch];

        float stddev;

        // Find mean
        for (int b = 0; b < batch; b++)
        {
            float* preImPtr =
                preOutData->mat + b * preOutData->cols + ch * chSize;

            for (int e = 0; e < chSize; e++)
            {
                mean += preImPtr[e];
            }
        }

        mean /= sampleSize;

        // Find variance
        for (int b = 0; b < batch; b++)
        {
            float* preImPtr =
                preOutData->mat + b * preOutData->cols + ch * chSize;

            for (int e = 0; e < chSize; e++)
            {
                float tmp = preImPtr[e] - mean;
                var += tmp * tmp;
            }
        }

        var /= sampleSize;
        stddev = sqrt(var + CNN_BN_EPS);

        // Process batch normalization
        for (int b = 0; b < batch; b++)
        {
            float* preImPtr =
                preOutData->mat + b * preOutData->cols + ch * chSize;
            float* imPtr = outData->mat + b * outData->cols + ch * chSize;

            for (int e = 0; e < chSize; e++)
            {
                float src = preImPtr[e];
                float out = (src - mean) / stddev;
                imPtr[e] = out * gamma + beta;
            }
        }

        // Assign values
        layerPtr->saveMean.mat[ch] = mean;
        layerPtr->saveVar.mat[ch] = var;

        layerPtr->runMean.mat[ch] =
            layerPtr->runMean.mat[ch] * (1.0 - expAvgFactor) +
            mean * expAvgFactor;
        layerPtr->runVar.mat[ch] =
            layerPtr->runVar.mat[ch] * (1.0 - expAvgFactor) +
            var * expAvgFactor;
    }

#endif
}

static inline void cnn_backward_bn(union CNN_LAYER* layerRef,
                                   struct CNN_CONFIG* cfgRef, int layerIndex)
{
    struct CNN_LAYER_BN* layerPtr = &layerRef[layerIndex].bn;

    struct CNN_MAT* outData = &layerPtr->outMat.data;
    struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

#ifdef CNN_WITH_CUDA
    float alpha = 1.0;
    float betaParam = 1.0;
    float betaGrad = 0.0;

    cnn_assert_cudnn(cudnnBatchNormalizationBackward(
        cnnInit.cudnnHandle, CUDNN_BATCHNORM_SPATIAL,  //
        &alpha, &betaGrad, &alpha, &betaParam,         //
        layerPtr->srcTen, preOutData->mat,             //
        layerPtr->srcTen, outData->grad,               //
        layerPtr->srcTen, preOutData->grad,            //
        layerPtr->bnTen, layerPtr->bnScale.mat, layerPtr->bnScale.grad,
        layerPtr->bnBias.grad,  //
        CNN_BN_EPS,             //
        layerPtr->saveMean.mat, layerPtr->saveVar.mat));

#else
    // Cache
    int channels = layerPtr->outMat.channel;
    int chSize = layerPtr->outMat.width * layerPtr->outMat.height;
    int batch = cfgRef->batch;

    float sampleSize = batch * chSize;

    for (int ch = 0; ch < channels; ch++)
    {
        float tmp;

        float rGrad = 0;
        float bGrad = 0;

        float varGrad = 0;
        float meanGrad = 0;

        float mean = layerPtr->saveMean.mat[ch];
        float var = layerPtr->saveVar.mat[ch];

        float stddev = sqrt(var + CNN_BN_EPS);

        float gamma = layerPtr->bnScale.mat[ch];

        // Find gamma, beta gradient
        for (int b = 0; b < batch; b++)
        {
            float* preImPtr =
                preOutData->mat + b * preOutData->cols + ch * chSize;
            float* gradPtr = outData->grad + b * outData->cols + ch * chSize;

            for (int e = 0; e < chSize; e++)
            {
                float srcNorm = (preImPtr[e] - mean) / stddev;
                float gradIn = gradPtr[e];

                rGrad += srcNorm * gradIn;
                bGrad += gradIn;
            }
        }

        layerPtr->bnScale.grad[ch] += rGrad;
        layerPtr->bnBias.grad[ch] += bGrad;

        // Find layer gradient
        if (layerIndex > 1)
        {
            // Find mean, variance gradient
            for (int b = 0; b < batch; b++)
            {
                float* preImPtr =
                    preOutData->mat + b * preOutData->cols + ch * chSize;
                float* gradPtr =
                    outData->grad + b * outData->cols + ch * chSize;

                for (int e = 0; e < chSize; e++)
                {
                    varGrad += gradPtr[e] * gamma * (preImPtr[e] - mean);
                }
            }

            varGrad = varGrad * -0.5 * pow(stddev, -3);

            // Find mean gradient
            tmp = 0;
            for (int b = 0; b < batch; b++)
            {
                float* preImPtr =
                    preOutData->mat + b * preOutData->cols + ch * chSize;
                float* gradPtr =
                    outData->grad + b * outData->cols + ch * chSize;

                for (int e = 0; e < chSize; e++)
                {
                    meanGrad += gradPtr[e] * gamma * -1.0 / stddev;
                    tmp += varGrad * -2.0 * (preImPtr[e] - mean);
                }
            }

            meanGrad += tmp / sampleSize;

            // Find layer gradient
            for (int b = 0; b < batch; b++)
            {
                float* preImPtr =
                    preOutData->mat + b * preOutData->cols + ch * chSize;
                float* preGradPtr =
                    preOutData->grad + b * preOutData->cols + ch * chSize;
                float* gradPtr =
                    outData->grad + b * outData->cols + ch * chSize;

                for (int e = 0; e < chSize; e++)
                {
                    preGradPtr[e] =
                        gradPtr[e] * gamma / stddev +
                        (varGrad * 2.0 * (preImPtr[e] - mean) + meanGrad) /
                            sampleSize;
                }
            }
        }
    }

#endif
}

#endif
