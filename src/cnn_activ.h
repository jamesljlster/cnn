#ifndef __CNN_ACTIV_H__
#define __CNN_ACTIV_H__

#include <cblas.h>

#include "cnn.h"
#include "cnn_builtin_math.h"
#include "cnn_macro.h"
#include "cnn_types.h"

#ifdef CNN_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cnn_builtin_math_cu.h"
#include "cnn_init.h"
#endif

static inline void cnn_forward_activ(union CNN_LAYER* layerRef,
                                     struct CNN_CONFIG* cfgRef, int layerIndex)
{
    cnn_activ_t id = cfgRef->layerCfg[layerIndex].activ.id;

    if (id > 0)
    {
        float* srcPtr = layerRef[layerIndex - 1].outMat.data.mat;
        float* dstPtr = layerRef[layerIndex].outMat.data.mat;
        float* bufPtr = layerRef[layerIndex].activ.buf.mat;

        cnn_activ_list[id](
            dstPtr, srcPtr,
            cfgRef->batch * layerRef[layerIndex].outMat.data.cols, bufPtr);
    }
    else
    {
#ifdef CNN_WITH_CUDA
        float alpha = 1.0;
        float beta = 0.0;

        struct CNN_LAYER_ACTIV* layerPtr = &layerRef[layerIndex].activ;

        struct CNN_MAT* outData = &layerPtr->outMat.data;
        struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

        cnn_assert_cudnn(cudnnSoftmaxForward(                 //
            cnnInit.cudnnHandle,                              //
            CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,  //
            &alpha,                                           //
            layerPtr->ten, preOutData->mat,                   //
            &beta,                                            //
            layerPtr->ten, outData->mat));
#else
        for (int j = 0; j < cfgRef->batch; j++)
        {
            int srcShift = j * layerRef[layerIndex - 1].outMat.data.cols;
            int dstShift = j * layerRef[layerIndex].outMat.data.cols;

            float* srcPtr = layerRef[layerIndex - 1].outMat.data.mat + srcShift;
            float* dstPtr = layerRef[layerIndex].outMat.data.mat + dstShift;
            float* bufPtr = layerRef[layerIndex].activ.buf.mat + dstShift;

            cnn_activ_list[id](dstPtr, srcPtr,
                               layerRef[layerIndex].outMat.data.cols, bufPtr);
        }
#endif
    }
}

static inline void cnn_backward_activ(union CNN_LAYER* layerRef,
                                      struct CNN_CONFIG* cfgRef, int layerIndex)
{
    cnn_activ_t id = cfgRef->layerCfg[layerIndex].activ.id;

    if (layerIndex > 1)
    {
        if (id > 0)
        {
            float* srcPtr = layerRef[layerIndex - 1].outMat.data.mat;
            float* dstPtr = layerRef[layerIndex].activ.gradMat.mat;

            float* gradIn = layerRef[layerIndex].outMat.data.grad;
            float* gradOut = layerRef[layerIndex - 1].outMat.data.grad;

            int size = cfgRef->batch * layerRef[layerIndex].outMat.data.cols;

            // Find gradient matrix
            cnn_activ_grad_list[cfgRef->layerCfg[layerIndex].activ.id](
                dstPtr, srcPtr, size, layerRef[layerIndex].outMat.data.mat);

            // Find layer gradient
#ifdef CNN_WITH_CUDA
            cnn_elemwise_product_gpu(gradOut, dstPtr, gradIn, size);
#else
#pragma omp parallel for shared(gradOut, dstPtr, gradIn)
            for (int k = 0; k < size; k++)
            {
                gradOut[k] = dstPtr[k] * gradIn[k];
            }
#endif
        }
        else
        {
#ifdef CNN_WITH_CUDA
            float alpha = 1.0;
            float beta = 0.0;

            struct CNN_LAYER_ACTIV* layerPtr = &layerRef[layerIndex].activ;

            struct CNN_MAT* outData = &layerPtr->outMat.data;
            struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

            cnn_assert_cudnn(cudnnSoftmaxBackward(                //
                cnnInit.cudnnHandle,                              //
                CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,  //
                &alpha,                                           //
                layerPtr->ten, outData->mat,                      //
                layerPtr->ten, outData->grad,                     //
                &beta,                                            //
                layerPtr->ten, preOutData->grad));

#else
            for (int j = 0; j < cfgRef->batch; j++)
            {
                int srcShift;
                int dstShift;

                float* srcPtr;
                float* dstPtr;

                srcShift = j * layerRef[layerIndex - 1].outMat.data.cols;
                if (cfgRef->layerCfg[layerIndex].activ.id == CNN_SOFTMAX)
                {
                    dstShift = srcShift * layerRef[layerIndex].outMat.data.cols;
                }
                else
                {
                    dstShift = srcShift;
                }

                srcPtr = layerRef[layerIndex - 1].outMat.data.mat + srcShift;
                dstPtr = layerRef[layerIndex].activ.gradMat.mat + dstShift;

                // Find gradient matrix
                cnn_activ_grad_list[cfgRef->layerCfg[layerIndex].activ.id](
                    dstPtr, srcPtr, layerRef[layerIndex].outMat.data.cols,
                    layerRef[layerIndex].outMat.data.mat + srcShift);

                // Find layer gradient
                if (cfgRef->layerCfg[layerIndex].activ.id == CNN_SOFTMAX)
                {
#ifdef CNN_WITH_CUDA
                    float alpha = 1.0;
                    float beta = 0.0;
                    cublasSgemm(
                        cnnInit.blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        layerRef[layerIndex].outMat.data.cols, 1,
                        layerRef[layerIndex].outMat.data.cols, &alpha, dstPtr,
                        layerRef[layerIndex].activ.gradMat.cols,
                        layerRef[layerIndex].outMat.data.grad + srcShift,
                        layerRef[layerIndex].outMat.data.cols, &beta,
                        layerRef[layerIndex - 1].outMat.data.grad + srcShift,
                        layerRef[layerIndex - 1].outMat.data.cols);
#else
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, 1,
                        layerRef[layerIndex - 1].outMat.data.cols,
                        layerRef[layerIndex].outMat.data.cols, 1.0,
                        &layerRef[layerIndex].outMat.data.grad[srcShift],
                        layerRef[layerIndex].outMat.data.cols, dstPtr,
                        layerRef[layerIndex].activ.gradMat.cols, 0.0,
                        &layerRef[layerIndex - 1].outMat.data.grad[srcShift],
                        layerRef[layerIndex - 1].outMat.data.cols);
#endif
                }
                else
                {
#ifdef CNN_WITH_CUDA
                    cnn_elemwise_product_gpu(
                        layerRef[layerIndex - 1].outMat.data.grad + srcShift,
                        dstPtr,
                        layerRef[layerIndex].outMat.data.grad + srcShift,
                        layerRef[layerIndex - 1].outMat.data.cols);
#else
                    for (int k = 0;
                         k < layerRef[layerIndex - 1].outMat.data.cols; k++)
                    {
                        layerRef[layerIndex - 1]
                            .outMat.data.grad[srcShift + k] =
                            dstPtr[k] *
                            layerRef[layerIndex].outMat.data.grad[srcShift + k];
                    }
#endif
                }
            }
#endif
        }
    }
}

#endif
