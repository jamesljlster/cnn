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
    // Cache
    struct CNN_LAYER_ACTIV* layerPtr = &layerRef[layerIndex].activ;

    struct CNN_MAT* outData = &layerPtr->outMat.data;
    struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

    cnn_activ_t id = cfgRef->layerCfg[layerIndex].activ.id;

    if (id > 0)
    {
        cnn_activ_list[id](outData->mat, preOutData->mat,
                           outData->rows * outData->cols, layerPtr->buf.mat);
    }
    else
    {
#ifdef CNN_WITH_CUDA
        float alpha = 1.0;
        float beta = 0.0;

        cnn_assert_cudnn(cudnnSoftmaxForward(                     //
            cnnInit.cudnnHandle,                                  //
            CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,  //
            &alpha,                                               //
            layerPtr->ten, preOutData->mat,                       //
            &beta,                                                //
            layerPtr->ten, outData->mat));
#else
        for (int j = 0; j < cfgRef->batch; j++)
        {
            int srcShift = j * preOutData->cols;
            int dstShift = j * outData->cols;

            float* srcPtr = preOutData->mat + srcShift;
            float* dstPtr = outData->mat + dstShift;
            float* bufPtr = layerPtr->buf.mat + dstShift;

            cnn_activ_list[id](dstPtr, srcPtr, outData->cols, bufPtr);
        }
#endif
    }
}

static inline void cnn_backward_activ(union CNN_LAYER* layerRef,
                                      struct CNN_CONFIG* cfgRef, int layerIndex)
{
    // Cache
    struct CNN_LAYER_ACTIV* layerPtr = &layerRef[layerIndex].activ;

    struct CNN_MAT* outData = &layerPtr->outMat.data;
    struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

    cnn_activ_t id = cfgRef->layerCfg[layerIndex].activ.id;

    // Find layer gradient
    if (layerIndex > 1)
    {
        if (id > 0)
        {
            cnn_activ_grad_list[id](
                preOutData->grad, outData->grad, preOutData->mat,
                outData->rows * outData->cols, outData->mat, layerPtr->buf.mat);
        }
        else
        {
#ifdef CNN_WITH_CUDA
            float alpha = 1.0;
            float beta = 0.0;

            cnn_assert_cudnn(cudnnSoftmaxBackward(                    //
                cnnInit.cudnnHandle,                                  //
                CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,  //
                &alpha,                                               //
                layerPtr->ten, outData->mat,                          //
                layerPtr->ten, outData->grad,                         //
                &beta,                                                //
                layerPtr->ten, preOutData->grad));

#else
            for (int j = 0; j < cfgRef->batch; j++)
            {
                int shift = j * outData->cols;

                // Gradient calculation
                cnn_activ_grad_list[id](
                    preOutData->grad + shift, outData->grad + shift,
                    preOutData->mat + shift, outData->cols,
                    outData->mat + shift, layerPtr->buf.mat);
            }
#endif
        }
    }
}

#endif
