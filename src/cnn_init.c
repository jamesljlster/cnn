#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cnn.h"
#include "cnn_init.h"
#include "cnn_private.h"

#ifdef CNN_WITH_CUDA
#include <assert.h>
#include <cuda_runtime.h>
#endif

#define CNN_M_PI 3.14159265359

struct CNN_INIT cnnInit = {0};

int cnn_init()
{
    int ret = CNN_NO_ERROR;

#ifdef CNN_WITH_CUDA
    cudaError_t cuRet;
#endif

    // Checking
    if (cnnInit.inited)
    {
        goto RET;
    }

    // Initial random seed
    cnnInit.randSeed = time(NULL);
    srand(cnnInit.randSeed);

#ifdef CNN_WITH_CUDA
    // Initial cublas
    cuRet = cublasCreate(&cnnInit.blasHandle);
    if (cuRet != cudaSuccess)
    {
        ret = CNN_CUDA_RUNTIME_ERROR;
        goto RET;
    }

    // Initial cuDNN
    cnn_run_cudnn(cudnnCreate(&cnnInit.cudnnHandle), ret, RET);

#endif

    // Assign value
    cnnInit.inited = 1;

RET:
    return ret;
}

void cnn_deinit()
{
    if (cnnInit.inited)
    {
#ifdef CNN_WITH_CUDA
        // Destroy cublas
        cublasDestroy(cnnInit.blasHandle);

        // Destroy cuDNN
        cudnnDestroy(cnnInit.cudnnHandle);

        // Free workspace
        cnn_free_cu(cnnInit.wsData);
#endif

        // Reset memory
        memset(&cnnInit, 0, sizeof(struct CNN_INIT));
    }
}

float cnn_normal_distribution(struct CNN_BOX_MULLER* bmPtr, double mean,
                              double stddev)
{
    double dPI = 2 * CNN_M_PI;
    double u0, u1;
    double z0, z1;
    double calcTmp;

    // Inverse saved
    bmPtr->saved = !(bmPtr->saved);

    if (!bmPtr->saved)
    {
        return bmPtr->val * stddev + mean;
    }

    // Generate
    do
    {
        u0 = (double)rand() / (double)RAND_MAX;
        u1 = (double)rand() / (double)RAND_MAX;
        calcTmp = u0 * u0 + u1 * u1;
    } while (calcTmp > 1.0 || calcTmp <= 0.0);

    z0 = sqrt(-2.0 * log(u0)) * cos(dPI * u1);
    z1 = sqrt(-2.0 * log(u0)) * sin(dPI * u1);
    bmPtr->val = z1;

    return z0 * stddev + mean;
}

float cnn_xavier_init(struct CNN_BOX_MULLER* bmPtr, int inSize, int outSize)
{
    double var;

    // Xavier initialization
    var = 2.0 / (double)(inSize + outSize);

    return cnn_normal_distribution(bmPtr, 0.0, sqrt(var));
}

void cnn_rand_network(cnn_t cnn)
{
    int i, j;
    size_t size;
    struct CNN_CONFIG* cfgRef;

    struct CNN_BOX_MULLER bm;

#ifdef CNN_WITH_CUDA
    int ret;
    float* tmpVec = NULL;
#endif

    // Get reference
    cfgRef = &cnn->cfg;

    // Rand network
    for (i = 1; i < cfgRef->layers; i++)
    {
        // Setup random method
        memset(&bm, 0, sizeof(struct CNN_BOX_MULLER));

        switch (cfgRef->layerCfg[i].type)
        {
            case CNN_LAYER_FC:
                // Random weight
                size = cnn->layerList[i].fc.weight.rows *
                       cnn->layerList[i].fc.weight.cols;

#ifdef CNN_WITH_CUDA
                // Buffer allocation
                cnn_alloc(tmpVec, size, float, ret, ERR);

                // Generate random distribution
                for (j = 0; j < size; j++)
                {
                    tmpVec[j] = cnn_xavier_init(
                        &bm, cnn->layerList[i - 1].outMat.data.cols,
                        cnn->layerList[i].outMat.data.cols);
                }

                // Copy memory
                cnn_run_cu(
                    cudaMemcpy(cnn->layerList[i].fc.weight.mat, tmpVec,
                               size * sizeof(float), cudaMemcpyHostToDevice),
                    ret, ERR);

                // Free buffer
                cnn_free(tmpVec);
                tmpVec = NULL;
#else
                // Generate random distribution
                for (j = 0; j < size; j++)
                {
                    cnn->layerList[i].fc.weight.mat[j] = cnn_xavier_init(
                        &bm, cnn->layerList[i - 1].outMat.data.cols,
                        cnn->layerList[i].outMat.data.cols);
                }
#endif

                // Zero bias
                size = cnn->layerList[i].fc.bias.rows *
                       cnn->layerList[i].fc.bias.cols;

#ifdef CNN_WITH_CUDA
                cnn_run_cu(cudaMemset(cnn->layerList[i].fc.bias.mat, 0,
                                      size * sizeof(float)),
                           ret, ERR);
#else
                memset(cnn->layerList[i].fc.bias.mat, 0, size * sizeof(float));
#endif

                break;

            case CNN_LAYER_CONV:
                // Random kernel
                size = cnn->layerList[i].conv.kernel.rows *
                       cnn->layerList[i].conv.kernel.cols;

#ifdef CNN_WITH_CUDA
                // Buffer allocation
                cnn_alloc(tmpVec, size, float, ret, ERR);

                // Generate random distribution
                for (j = 0; j < size; j++)
                {
                    tmpVec[j] = cnn_xavier_init(
                        &bm, cnn->layerList[i - 1].outMat.data.cols,
                        cnn->layerList[i].outMat.data.cols);
                }

                // Copy memory
                cnn_run_cu(
                    cudaMemcpy(cnn->layerList[i].conv.kernel.mat, tmpVec,
                               size * sizeof(float), cudaMemcpyHostToDevice),
                    ret, ERR);

                // Free buffer
                cnn_free(tmpVec);
                tmpVec = NULL;
#else
                // Generate random distribution
                for (j = 0; j < size; j++)
                {
                    cnn->layerList[i].conv.kernel.mat[j] = cnn_xavier_init(
                        &bm, cnn->layerList[i - 1].outMat.data.cols,
                        cnn->layerList[i].outMat.data.cols);
                }
#endif

                // Zero bias
#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
                size = cnn->layerList[i].conv.bias.rows *
                       cnn->layerList[i].conv.bias.cols;

#ifdef CNN_WITH_CUDA
                cnn_run_cu(cudaMemset(cnn->layerList[i].conv.bias.mat, 0,
                                      size * sizeof(float)),
                           ret, ERR);
#else
                memset(cnn->layerList[i].conv.bias.mat, 0,
                       size * sizeof(float));
#endif
#endif

                break;

            case CNN_LAYER_TEXT:
                // Random weight
                size = cnn->layerList[i].text.weight.rows *
                       cnn->layerList[i].text.weight.cols;

#ifdef CNN_WITH_CUDA
                // Buffer allocation
                cnn_alloc(tmpVec, size, float, ret, ERR);

                // Generate random distribution
                for (j = 0; j < size; j++)
                {
                    tmpVec[j] = cnn_xavier_init(
                        &bm, cnn->layerList[i - 1].outMat.data.cols,
                        cnn->layerList[i].outMat.data.cols);
                }

                // Copy memory
                cnn_run_cu(
                    cudaMemcpy(cnn->layerList[i].text.weight.mat, tmpVec,
                               size * sizeof(float), cudaMemcpyHostToDevice),
                    ret, ERR);

                // Free buffer
                cnn_free(tmpVec);
                tmpVec = NULL;
#else
                // Generate random distribution
                for (j = 0; j < size; j++)
                {
                    cnn->layerList[i].text.weight.mat[j] = cnn_xavier_init(
                        &bm, cnn->layerList[i - 1].outMat.data.cols,
                        cnn->layerList[i].outMat.data.cols);
                }
#endif

                // Zero bias
                size = cnn->layerList[i].text.bias.rows *
                       cnn->layerList[i].text.bias.cols;

#ifdef CNN_WITH_CUDA
                cnn_run_cu(cudaMemset(cnn->layerList[i].text.bias.mat, 0,
                                      size * sizeof(float)),
                           ret, ERR);
#else
                memset(cnn->layerList[i].text.bias.mat, 0,
                       size * sizeof(float));
#endif
                break;

            case CNN_LAYER_INPUT:
            case CNN_LAYER_ACTIV:
            case CNN_LAYER_POOL:
            case CNN_LAYER_DROP:
            case CNN_LAYER_BN:
                break;
        }
    }

#ifdef CNN_WITH_CUDA
    goto RET;

ERR:
    (void)ret;
    assert(!"Rumtime error!");

RET:
    cnn_free(tmpVec);
#endif
}

void cnn_zero_network(cnn_t cnn)
{
    int i;
    size_t size;
    struct CNN_CONFIG* cfgRef;

#ifdef CNN_WITH_CUDA
    int ret;
#endif

    // Get reference
    cfgRef = &cnn->cfg;

    // Rand network
    for (i = 1; i < cfgRef->layers; i++)
    {
        switch (cfgRef->layerCfg[i].type)
        {
            case CNN_LAYER_FC:
                // Zero weight
                size = cnn->layerList[i].fc.weight.rows *
                       cnn->layerList[i].fc.weight.cols;

#ifdef CNN_WITH_CUDA
                cnn_run_cu(cudaMemset(cnn->layerList[i].fc.weight.mat, 0,
                                      size * sizeof(float)),
                           ret, ERR);
#else
                memset(cnn->layerList[i].fc.weight.mat, 0,
                       size * sizeof(float));
#endif

                // Zero bias
                size = cnn->layerList[i].fc.bias.rows *
                       cnn->layerList[i].fc.bias.cols;

#ifdef CNN_WITH_CUDA
                cnn_run_cu(cudaMemset(cnn->layerList[i].fc.bias.mat, 0,
                                      size * sizeof(float)),
                           ret, ERR);
#else
                memset(cnn->layerList[i].fc.bias.mat, 0, size * sizeof(float));
#endif

                break;

            case CNN_LAYER_CONV:
                // Zero kernel
                size = cnn->layerList[i].conv.kernel.rows *
                       cnn->layerList[i].conv.kernel.cols;

#ifdef CNN_WITH_CUDA
                cnn_run_cu(cudaMemset(cnn->layerList[i].conv.kernel.mat, 0,
                                      size * sizeof(float)),
                           ret, ERR);
#else
                memset(cnn->layerList[i].conv.kernel.mat, 0,
                       size * sizeof(float));
#endif

                // Zero bias
#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
                size = cnn->layerList[i].conv.bias.rows *
                       cnn->layerList[i].conv.bias.cols;

#ifdef CNN_WITH_CUDA
                cnn_run_cu(cudaMemset(cnn->layerList[i].conv.bias.mat, 0,
                                      size * sizeof(float)),
                           ret, ERR);
#else
                memset(cnn->layerList[i].conv.bias.mat, 0,
                       size * sizeof(float));
#endif
#endif

                break;

            case CNN_LAYER_TEXT:
                // Zero weight
                size = cnn->layerList[i].text.weight.rows *
                       cnn->layerList[i].text.weight.cols;

#ifdef CNN_WITH_CUDA
                cnn_run_cu(cudaMemset(cnn->layerList[i].text.weight.mat, 0,
                                      size * sizeof(float)),
                           ret, ERR);
#else
                memset(cnn->layerList[i].text.weight.mat, 0,
                       size * sizeof(float));
#endif

                // Zero bias
                size = cnn->layerList[i].text.bias.rows *
                       cnn->layerList[i].text.bias.cols;

#ifdef CNN_WITH_CUDA
                cnn_run_cu(cudaMemset(cnn->layerList[i].text.bias.mat, 0,
                                      size * sizeof(float)),
                           ret, ERR);
#else
                memset(cnn->layerList[i].text.bias.mat, 0,
                       size * sizeof(float));
#endif
                break;

            case CNN_LAYER_INPUT:
            case CNN_LAYER_ACTIV:
            case CNN_LAYER_POOL:
            case CNN_LAYER_DROP:
            case CNN_LAYER_BN:
                break;
        }
    }

#ifdef CNN_WITH_CUDA
    goto RET;

ERR:
    assert(!"Rumtime error!");

RET:
    (void)ret;
#endif
}

#ifdef CNN_WITH_CUDA
int cnn_cudnn_ws_alloc(void)
{
    int ret = CNN_NO_ERROR;

    // Check initialize status
    if (!cnnInit.inited)
    {
        ret = CNN_NOT_INITIALIZED;
        goto RET;
    }

    // Allocate cuDNN workspace
    if (cnnInit.wsSize)
    {
        if (cudaMalloc((void**)&cnnInit.wsData, cnnInit.wsSize) != cudaSuccess)
        {
            ret = CNN_MEM_FAILED;
            goto RET;
        }
    }

RET:
    return ret;
}
#endif
