#include <assert.h>
#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "cnn_calc.h"
#include "cnn_init.h"
#include "cnn_private.h"

inline void cnn_restrict(float* mat, int size, float limit)
{
    for (int __i = 0; __i < size; __i++)
    {
        mat[__i] = fminf(mat[__i], limit);
    }
}

void cnn_mat_update(struct CNN_MAT* matPtr, float lRate, float limit)
{
    int size = matPtr->rows * matPtr->cols;

    // Limit gradient and update weight
#ifdef CNN_WITH_CUDA
    cnn_fminf_gpu(matPtr->grad, matPtr->grad, size, limit);
    cublasSaxpy(cnnInit.blasHandle, size, &lRate, matPtr->grad, 1, matPtr->mat,
                1);
#else
    cnn_restrict(matPtr->grad, size, limit);
    cblas_saxpy(size, lRate, matPtr->grad, 1, matPtr->mat, 1);
#endif

    // Clear gradient
#ifdef CNN_WITH_CUDA
    cudaMemset
#else
    memset
#endif
        (matPtr->grad, 0, size * sizeof(float));
}

void cnn_update(cnn_t cnn, float lRate, float gradLimit)
{
    int i;
    struct CNN_CONFIG* cfgRef;
    union CNN_LAYER* layerRef;

    // Set reference
    layerRef = cnn->layerList;
    cfgRef = &cnn->cfg;

    // Update network and clear gradient
    for (i = cfgRef->layers - 1; i > 0; i--)
    {
        // Clear layer gradient
#ifdef CNN_WITH_CUDA
        cudaMemset
#else
        memset
#endif
            (layerRef[i].outMat.data.grad, 0,
             sizeof(float) * layerRef[i].outMat.data.rows *
                 layerRef[i].outMat.data.cols);

        switch (cfgRef->layerCfg[i].type)
        {
            // Fully connected
            case CNN_LAYER_FC:
                cnn_mat_update(&layerRef[i].fc.weight, lRate, gradLimit);
                cnn_mat_update(&layerRef[i].fc.bias, lRate, gradLimit);
                break;

            // Convolution
            case CNN_LAYER_CONV:
                cnn_mat_update(&layerRef[i].conv.kernel, lRate, gradLimit);

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
                cnn_mat_update(&layerRef[i].conv.bias, lRate, gradLimit);
#endif
                break;

            // Batch normalization
            case CNN_LAYER_BN:
                cnn_mat_update(&layerRef[i].bn.bnVar, lRate, gradLimit);
                break;

            // Texture
            case CNN_LAYER_TEXT:
                cnn_mat_update(&layerRef[i].text.weight, lRate, gradLimit);
                cnn_mat_update(&layerRef[i].text.bias, lRate, gradLimit);
                cnn_mat_update(&layerRef[i].text.alpha, lRate, gradLimit);
                break;

            case CNN_LAYER_INPUT:
            case CNN_LAYER_ACTIV:
            case CNN_LAYER_POOL:
            case CNN_LAYER_DROP:
                break;
        }
    }
}

static inline void cnn_backward_kernel(cnn_t cnn, float* errGrad)
{
    int i;

    struct CNN_CONFIG* cfgRef;
    union CNN_LAYER* layerRef;

    // Set reference
    layerRef = cnn->layerList;
    cfgRef = &cnn->cfg;

    // Backpropagation
    for (i = cfgRef->layers - 1; i > 0; i--)
    {
        switch (cfgRef->layerCfg[i].type)
        {
            // Fully connected
            case CNN_LAYER_FC:
                cnn_backward_fc(layerRef, cfgRef, i);
                break;

            // Activation function
            case CNN_LAYER_ACTIV:
                cnn_backward_activ(layerRef, cfgRef, i);
                break;

            // Convolution
            case CNN_LAYER_CONV:
                cnn_backward_conv(layerRef, cfgRef, i);
                break;

            // Pooling
            case CNN_LAYER_POOL:
                cnn_backward_pool(layerRef, cfgRef, i);
                break;

            // Dropout
            case CNN_LAYER_DROP:
                cnn_backward_drop(layerRef, cfgRef, i);
                break;

            // Batch normalization
            case CNN_LAYER_BN:
                cnn_backward_bn(layerRef, cfgRef, i);
                break;

            // Texture
            case CNN_LAYER_TEXT:
                cnn_backward_text(layerRef, cfgRef, i);
                break;

            default:
                assert(!"Invalid layer type");
        }
    }
}

void cnn_backward(cnn_t cnn, float* errGrad)
{
    int size;

    struct CNN_CONFIG* cfgRef;
    union CNN_LAYER* layerRef;

    // Set reference
    layerRef = cnn->layerList;
    cfgRef = &cnn->cfg;

    // Copy gradient vector
    size = sizeof(float) * layerRef[cfgRef->layers - 1].outMat.data.rows *
           layerRef[cfgRef->layers - 1].outMat.data.cols;

#ifdef CNN_WITH_CUDA
    cudaMemcpy(layerRef[cfgRef->layers - 1].outMat.data.grad, errGrad, size,
               cudaMemcpyHostToDevice);
#else
    memcpy(layerRef[cfgRef->layers - 1].outMat.data.grad, errGrad, size);
#endif

    // Backpropagation
    cnn_backward_kernel(cnn, errGrad);
}

#ifdef CNN_WITH_CUDA
void cnn_backward_gpu(cnn_t cnn, float* errGrad)
{
    int size;

    struct CNN_CONFIG* cfgRef;
    union CNN_LAYER* layerRef;

    // Set reference
    layerRef = cnn->layerList;
    cfgRef = &cnn->cfg;

    // Copy gradient vector
    size = sizeof(float) * layerRef[cfgRef->layers - 1].outMat.data.rows *
           layerRef[cfgRef->layers - 1].outMat.data.cols;
    cudaMemcpy(layerRef[cfgRef->layers - 1].outMat.data.grad, errGrad, size,
               cudaMemcpyDeviceToDevice);

    // Backpropagation
    cnn_backward_kernel(cnn, errGrad);
}
#endif

static inline void cnn_forward_kernel(cnn_t cnn, float* inputMat,
                                      float* outputMat)
{
    int i;

    struct CNN_CONFIG* cfgRef;
    union CNN_LAYER* layerRef;

    // Set reference
    layerRef = cnn->layerList;
    cfgRef = &cnn->cfg;

    // Forward computation
    for (i = 1; i < cfgRef->layers; i++)
    {
        switch (cfgRef->layerCfg[i].type)
        {
            // Fully connected
            case CNN_LAYER_FC:
                cnn_forward_fc(layerRef, cfgRef, i);
                break;

            // Activation function
            case CNN_LAYER_ACTIV:
                cnn_forward_activ(layerRef, cfgRef, i);
                break;

            // Convolution
            case CNN_LAYER_CONV:
                cnn_forward_conv(layerRef, cfgRef, i);
                break;

            // Pooling
            case CNN_LAYER_POOL:
                cnn_forward_pool(layerRef, cfgRef, i);
                break;

            // Dropout
            case CNN_LAYER_DROP:
                // if (cnn->dropEnable)
                if (cnn->opMode == CNN_OPMODE_TRAIN)
                {
                    cnn_forward_drop(layerRef, cfgRef, i);
                }
                else
                {
#ifdef CNN_WITH_CUDA
                    cudaMemcpy(layerRef[i].outMat.data.mat,
                               layerRef[i - 1].outMat.data.mat,
                               sizeof(float) * layerRef[i].outMat.data.rows *
                                   layerRef[i].outMat.data.cols,
                               cudaMemcpyDeviceToDevice);
#else
                    memcpy(layerRef[i].outMat.data.mat,
                           layerRef[i - 1].outMat.data.mat,
                           sizeof(float) * layerRef[i].outMat.data.rows *
                               layerRef[i].outMat.data.cols);
#endif
                }

                break;

            // Batch normalization
            case CNN_LAYER_BN:
                cnn_forward_bn(layerRef, cfgRef, i);
                break;

            // Texture
            case CNN_LAYER_TEXT:
                cnn_forward_text(layerRef, cfgRef, i);
                break;

            default:
                assert(!"Invalid layer type");
        }
    }
}

void cnn_forward(cnn_t cnn, float* inputMat, float* outputMat)
{
    int size;

    struct CNN_CONFIG* cfgRef;
    union CNN_LAYER* layerRef;

    // Set reference
    layerRef = cnn->layerList;
    cfgRef = &cnn->cfg;

    // Copy input
    size = sizeof(float) * layerRef[0].outMat.data.rows *
           layerRef[0].outMat.data.cols;
#ifdef CNN_WITH_CUDA
    cudaMemcpy(layerRef[0].outMat.data.mat, inputMat, size,
               cudaMemcpyHostToDevice);
#else
    memcpy(layerRef[0].outMat.data.mat, inputMat, size);
#endif

    // Forward computation
    cnn_forward_kernel(cnn, inputMat, outputMat);

    // Copy output
    if (outputMat != NULL)
    {
        size = sizeof(float) * layerRef[cfgRef->layers - 1].outMat.data.rows *
               layerRef[cfgRef->layers - 1].outMat.data.cols;
#ifdef CNN_WITH_CUDA
        cudaMemcpy(outputMat, layerRef[cfgRef->layers - 1].outMat.data.mat,
                   size, cudaMemcpyDeviceToHost);
#else
        memcpy(outputMat, layerRef[cfgRef->layers - 1].outMat.data.mat, size);
#endif
    }
#ifdef CNN_WITH_CUDA
    else
    {
        cudaDeviceSynchronize();
    }
#endif
}

#ifdef CNN_WITH_CUDA
void cnn_forward_gpu(cnn_t cnn, float* inputMat, float* outputMat)
{
    int size;

    struct CNN_CONFIG* cfgRef;
    union CNN_LAYER* layerRef;

    // Set reference
    layerRef = cnn->layerList;
    cfgRef = &cnn->cfg;

    // Copy input
    size = sizeof(float) * layerRef[0].outMat.data.rows *
           layerRef[0].outMat.data.cols;
    cudaMemcpy(layerRef[0].outMat.data.mat, inputMat, size,
               cudaMemcpyDeviceToDevice);

    // Forward computation
    cnn_forward_kernel(cnn, inputMat, outputMat);

    // Copy output
    if (outputMat != NULL)
    {
        size = sizeof(float) * layerRef[cfgRef->layers - 1].outMat.data.rows *
               layerRef[cfgRef->layers - 1].outMat.data.cols;
        cudaMemcpy(outputMat, layerRef[cfgRef->layers - 1].outMat.data.mat,
                   size, cudaMemcpyDeviceToDevice);
    }
    else
    {
        cudaDeviceSynchronize();
    }
}
#endif
