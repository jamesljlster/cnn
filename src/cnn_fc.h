#ifndef __CNN_FC_H__
#define __CNN_FC_H__

#include <cblas.h>

#include "cnn_types.h"

#ifdef CNN_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cnn_init.h"
#endif

static inline void cnn_forward_fc(union CNN_LAYER* layerRef,
                                  struct CNN_CONFIG* cfgRef, int layerIndex)
{
#ifdef CNN_WITH_CUDA
    float alpha = 1.0;
    float beta = 0.0;

    struct CNN_LAYER_FC* layerPtr = &layerRef[layerIndex].fc;

    struct CNN_MAT* wPtr = &layerPtr->weight;
    struct CNN_MAT* outData = &layerPtr->outMat.data;
    struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

    // Weight matrix multiplication
    cnn_assert_cu(cublasSgemm(                           //
        cnnInit.blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,    //
        outData->cols, cfgRef->batch, preOutData->cols,  //
        &alpha,                                          //
        wPtr->mat, wPtr->cols,                           //
        preOutData->mat, preOutData->cols,               //
        &beta,                                           //
        outData->mat, outData->cols));

    // Add bias
    beta = 1.0;
    cnn_assert_cudnn(cudnnAddTensor(cnnInit.cudnnHandle,                    //
                                    &alpha,                                 //
                                    layerPtr->biasTen, layerPtr->bias.mat,  //
                                    &beta,                                  //
                                    layerPtr->dstTen, outData->mat));
#else
    // Weight matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                layerRef[layerIndex - 1].outMat.data.rows,
                layerRef[layerIndex].outMat.data.cols,
                layerRef[layerIndex - 1].outMat.data.cols, 1.0,
                layerRef[layerIndex - 1].outMat.data.mat,
                layerRef[layerIndex - 1].outMat.data.cols,
                layerRef[layerIndex].fc.weight.mat,
                layerRef[layerIndex].fc.weight.cols, 0.0,
                layerRef[layerIndex].outMat.data.mat,
                layerRef[layerIndex].outMat.data.cols);

    // Add bias
    for (int j = 0; j < cfgRef->batch; j++)
    {
        int dstIndex = j * layerRef[layerIndex].outMat.data.cols;
        cblas_saxpy(layerRef[layerIndex].fc.bias.cols, 1.0,
                    layerRef[layerIndex].fc.bias.mat, 1,
                    &layerRef[layerIndex].outMat.data.mat[dstIndex], 1);
    }
#endif
}

static inline void cnn_backward_fc(union CNN_LAYER* layerRef,
                                   struct CNN_CONFIG* cfgRef, int layerIndex)
{
#ifdef CNN_WITH_CUDA
    float alpha = 1.0;
    float beta = 1.0;

    struct CNN_LAYER_FC* layerPtr = &layerRef[layerIndex].fc;

    struct CNN_MAT* wPtr = &layerPtr->weight;
    struct CNN_MAT* outData = &layerPtr->outMat.data;
    struct CNN_MAT* preOutData = &layerRef[layerIndex - 1].outMat.data;

    // Sum weight gradient matrix
    cnn_assert_cu(cublasSgemm(cnnInit.blasHandle, CUBLAS_OP_N, CUBLAS_OP_T,  //
                              wPtr->cols, wPtr->rows, cfgRef->batch,         //
                              &alpha,                                        //
                              outData->grad, outData->cols,                  //
                              preOutData->mat, preOutData->cols,             //
                              &beta,                                         //
                              wPtr->grad, wPtr->cols));

    // Sum bias gradient matrix
    if (cfgRef->batch > 1)
    {
        cnn_assert_cudnn(
            cudnnReduceTensor(cnnInit.cudnnHandle, layerPtr->reduDesc,
                              layerPtr->indData, layerPtr->indSize,  //
                              cnnInit.wsData, cnnInit.wsSize,        //
                              &alpha,                                //
                              layerPtr->dstTen, outData->grad,       //
                              &beta,                                 //
                              layerPtr->biasTen, layerPtr->bias.grad));
    }
    else
    {
        cublasSaxpy(cnnInit.blasHandle, layerPtr->bias.cols, &alpha,
                    outData->grad, 1, layerPtr->bias.grad, 1);
    }
#else
    // Sum weight gradient matrix
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                layerRef[layerIndex].fc.weight.rows,
                layerRef[layerIndex].fc.weight.cols, cfgRef->batch, 1.0,
                layerRef[layerIndex - 1].outMat.data.mat,
                layerRef[layerIndex - 1].outMat.data.cols,
                layerRef[layerIndex].outMat.data.grad,
                layerRef[layerIndex].outMat.data.cols, 1.0,
                layerRef[layerIndex].fc.weight.grad,
                layerRef[layerIndex].fc.weight.cols);

    // Sum bias gradient matrix
    for (int j = 0; j < cfgRef->batch; j++)
    {
        int srcShift = j * layerRef[layerIndex].outMat.data.cols;
        cblas_saxpy(layerRef[layerIndex].fc.bias.cols, 1.0,
                    &layerRef[layerIndex].outMat.data.grad[srcShift], 1,
                    layerRef[layerIndex].fc.bias.grad, 1);
    }
#endif

    // Find layer gradient
    if (layerIndex > 1)
    {
#ifdef CNN_WITH_CUDA
        beta = 0.0;
        cublasSgemm(cnnInit.blasHandle, CUBLAS_OP_T, CUBLAS_OP_N,  //
                    wPtr->rows, cfgRef->batch, wPtr->cols,         //
                    &alpha,                                        //
                    wPtr->mat, wPtr->cols,                         //
                    outData->grad, outData->cols,                  //
                    &beta,                                         //
                    preOutData->grad, preOutData->cols);
#else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    layerRef[layerIndex - 1].outMat.data.rows,
                    layerRef[layerIndex - 1].outMat.data.cols,
                    layerRef[layerIndex].outMat.data.cols, 1.0,
                    layerRef[layerIndex].outMat.data.grad,
                    layerRef[layerIndex].outMat.data.cols,
                    layerRef[layerIndex].fc.weight.mat,
                    layerRef[layerIndex].fc.weight.cols, 0.0,
                    layerRef[layerIndex - 1].outMat.data.grad,
                    layerRef[layerIndex - 1].outMat.data.cols);
#endif
    }
}

#endif
