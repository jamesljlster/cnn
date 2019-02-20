#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "cnn.h"
#include "cnn_builtin_math.h"
#include "cnn_calc.h"
#include "cnn_init.h"
#include "cnn_private.h"

int cnn_network_alloc(struct CNN* cnn)
{
    int i;
    int ret = CNN_NO_ERROR;
    int tmpWidth, tmpHeight, tmpChannel;

    struct CNN_CONFIG* cfg;

    // Check initialize status
    if (!cnnInit.inited)
    {
        ret = CNN_NOT_INITIALIZED;
        goto RET;
    }

    // Set reference
    cfg = &cnn->cfg;

    // Memory allocation
    cnn_alloc(cnn->layerList, cfg->layers, union CNN_LAYER, ret, ERR);

    // Allocate CNN layers
    tmpWidth = cfg->width;
    tmpHeight = cfg->height;
    tmpChannel = cfg->channel;
    for (i = 0; i < cfg->layers; i++)
    {
        switch (cfg->layerCfg[i].type)
        {
            case CNN_LAYER_INPUT:
                cnn_run(cnn_layer_input_alloc(
                            &cnn->layerList[i].input,
                            (struct CNN_CONFIG_LAYER_INPUT*)&cfg->layerCfg[i],
                            tmpWidth, tmpHeight, tmpChannel, cfg->batch),
                        ret, ERR);
                break;

            case CNN_LAYER_FC:
                cnn_run(cnn_layer_fc_alloc(
                            &cnn->layerList[i].fc,
                            (struct CNN_CONFIG_LAYER_FC*)&cfg->layerCfg[i],
                            tmpWidth, tmpHeight, tmpChannel, cfg->batch),
                        ret, ERR);
                break;

            case CNN_LAYER_ACTIV:
                cnn_run(cnn_layer_activ_alloc(
                            &cnn->layerList[i].activ,
                            (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[i],
                            tmpWidth, tmpHeight, tmpChannel, cfg->batch),
                        ret, ERR);
                break;

            case CNN_LAYER_CONV:
                cnn_run(cnn_layer_conv_alloc(
                            &cnn->layerList[i].conv,
                            (struct CNN_CONFIG_LAYER_CONV*)&cfg->layerCfg[i],
                            tmpWidth, tmpHeight, tmpChannel, cfg->batch),
                        ret, ERR);
                break;

            case CNN_LAYER_POOL:
                cnn_run(cnn_layer_pool_alloc(
                            &cnn->layerList[i].pool,
                            (struct CNN_CONFIG_LAYER_POOL*)&cfg->layerCfg[i],
                            tmpWidth, tmpHeight, tmpChannel, cfg->batch),
                        ret, ERR);
                break;

            case CNN_LAYER_DROP:
                cnn_run(cnn_layer_drop_alloc(
                            &cnn->layerList[i].drop,
                            (struct CNN_CONFIG_LAYER_DROP*)&cfg->layerCfg[i],
                            tmpWidth, tmpHeight, tmpChannel, cfg->batch),
                        ret, ERR);
                break;

            case CNN_LAYER_BN:
                cnn_run(cnn_layer_bn_alloc(
                            &cnn->layerList[i].bn,
                            (struct CNN_CONFIG_LAYER_BN*)&cfg->layerCfg[i],
                            tmpWidth, tmpHeight, tmpChannel, cfg->batch),
                        ret, ERR);
                break;

            default:
                assert(!"Invalid layer type");
        }

        // Find layer output image size
        tmpWidth = cnn->layerList[i].outMat.width;
        tmpHeight = cnn->layerList[i].outMat.height;
        tmpChannel = cnn->layerList[i].outMat.channel;
    }

    goto RET;

ERR:
    cnn_network_delete(cnn);

RET:
    return ret;
}

int cnn_mat_alloc(struct CNN_MAT* matPtr, int rows, int cols, int needGrad)
{
    int ret = CNN_NO_ERROR;

    // Checking
    if (rows <= 0 || cols <= 0)
    {
        ret = CNN_INVALID_SHAPE;
        goto RET;
    }

    // Memory allocation
#ifdef CNN_WITH_CUDA
    cnn_alloc_cu(matPtr->mat, rows * cols, float, ret, ERR);
#else
    cnn_alloc(matPtr->mat, rows * cols, float, ret, ERR);
#endif

    if (needGrad > 0)
    {
#ifdef CNN_WITH_CUDA
        cnn_alloc_cu(matPtr->grad, rows * cols, float, ret, ERR);
#else
        cnn_alloc(matPtr->grad, rows * cols, float, ret, ERR);
#endif
    }
    else
    {
        matPtr->grad = NULL;
    }

    // Assign value
    matPtr->rows = rows;
    matPtr->cols = cols;

    goto RET;

ERR:
    cnn_mat_delete(matPtr);

RET:
    return ret;
}

int cnn_layer_input_alloc(struct CNN_LAYER_INPUT* layerPtr,
                          struct CNN_CONFIG_LAYER_INPUT* cfgPtr, int inWidth,
                          int inHeight, int inChannel, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;
    int outWidth, outHeight, outChannel;

    // Find output shape
    cnn_run(cnn_config_find_layer_outsize(&outWidth, &outHeight, &outChannel,
                                          inWidth, inHeight, inChannel,
                                          (union CNN_CONFIG_LAYER*)cfgPtr),
            ret, RET);

    // Find allocate size
    outRows = batch;
    outCols = inWidth * inHeight * inChannel;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 0), ret,
            ERR);

    // Assign value
    layerPtr->outMat.width = outWidth;
    layerPtr->outMat.height = outHeight;
    layerPtr->outMat.channel = outChannel;

    goto RET;

ERR:
    cnn_layer_input_delete(layerPtr);

RET:
    return ret;
}

int cnn_layer_drop_alloc(struct CNN_LAYER_DROP* layerPtr,
                         struct CNN_CONFIG_LAYER_DROP* cfgPtr, int inWidth,
                         int inHeight, int inChannel, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;
    int outWidth, outHeight, outChannel;

    // Find output shape
    cnn_run(cnn_config_find_layer_outsize(&outWidth, &outHeight, &outChannel,
                                          inWidth, inHeight, inChannel,
                                          (union CNN_CONFIG_LAYER*)cfgPtr),
            ret, RET);

    // Find allocate size
    outRows = batch;
    outCols = inWidth * inHeight * inChannel;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);
    cnn_alloc(layerPtr->mask, outRows * outCols, int, ret, ERR);

#ifdef CNN_WITH_CUDA
    cnn_alloc_cu(layerPtr->maskGpu, outRows * outCols, int, ret, ERR);
#endif

    // Assign value
    layerPtr->outMat.width = outWidth;
    layerPtr->outMat.height = outHeight;
    layerPtr->outMat.channel = outChannel;

    goto RET;

ERR:
    cnn_layer_drop_delete(layerPtr);

RET:
    return ret;
}

int cnn_layer_activ_alloc(struct CNN_LAYER_ACTIV* layerPtr,
                          struct CNN_CONFIG_LAYER_ACTIV* cfgPtr, int inWidth,
                          int inHeight, int inChannel, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;
    int gradRows, gradCols;
    int outWidth, outHeight, outChannel;

    // Find output shape
    cnn_run(cnn_config_find_layer_outsize(&outWidth, &outHeight, &outChannel,
                                          inWidth, inHeight, inChannel,
                                          (union CNN_CONFIG_LAYER*)cfgPtr),
            ret, RET);

    // Find allocate size
    outRows = batch;
    outCols = inWidth * inHeight * inChannel;

    gradRows = batch;
    gradCols = outCols;

    if (cfgPtr->id == CNN_SOFTMAX)
    {
        gradRows = batch * outCols;
    }

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->gradMat, gradRows, gradCols, 0), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->buf, outRows, outCols, 0), ret, ERR);

    // Assign value
    layerPtr->outMat.width = outWidth;
    layerPtr->outMat.height = outHeight;
    layerPtr->outMat.channel = outChannel;

    goto RET;

ERR:
    cnn_layer_activ_delete(layerPtr);

RET:
    return ret;
}

int cnn_layer_fc_alloc(struct CNN_LAYER_FC* layerPtr,
                       struct CNN_CONFIG_LAYER_FC* cfgPtr, int inWidth,
                       int inHeight, int inChannel, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;  // Output matrix size
    int wRows, wCols;      // Weight matrix size
    int bRows, bCols;      // Bias matrix size
    int outWidth, outHeight, outChannel;

    // Find output shape
    cnn_run(cnn_config_find_layer_outsize(&outWidth, &outHeight, &outChannel,
                                          inWidth, inHeight, inChannel,
                                          (union CNN_CONFIG_LAYER*)cfgPtr),
            ret, RET);

    // Find allocate size
    outRows = batch;
    outCols = cfgPtr->size;

    wRows = inWidth * inHeight * inChannel;
    wCols = cfgPtr->size;

    bRows = 1;
    bCols = cfgPtr->size;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->weight, wRows, wCols, 1), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->bias, bRows, bCols, 1), ret, ERR);

    // Assign value
    layerPtr->outMat.width = outWidth;
    layerPtr->outMat.height = outHeight;
    layerPtr->outMat.channel = outChannel;

    goto RET;

ERR:
    cnn_layer_fc_delete(layerPtr);

RET:
    return ret;
}

int cnn_layer_pool_alloc(struct CNN_LAYER_POOL* layerPtr,
                         struct CNN_CONFIG_LAYER_POOL* cfgPtr, int inWidth,
                         int inHeight, int inChannel, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;                 // Output matrix size
    int outWidth, outHeight, outChannel;  // Valid pooling output size

    // Find output image size
    cnn_run(cnn_config_find_layer_outsize(&outWidth, &outHeight, &outChannel,
                                          inWidth, inHeight, inChannel,
                                          (union CNN_CONFIG_LAYER*)cfgPtr),
            ret, RET);

    // Find allocate size
    outRows = batch;
    outCols = outWidth * outHeight * inChannel;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);

#ifdef CNN_WITH_CUDA
    cnn_alloc_cu(layerPtr->indexMat, outRows * outCols, int, ret, ERR);
#else
    cnn_alloc(layerPtr->indexMat, outRows * outCols, int, ret, ERR);
#endif

    // Assing value
    layerPtr->outMat.width = outWidth;
    layerPtr->outMat.height = outHeight;
    layerPtr->outMat.channel = outChannel;

    goto RET;

ERR:
    cnn_layer_pool_delete(layerPtr);

RET:
    return ret;
}

int cnn_layer_conv_alloc(struct CNN_LAYER_CONV* layerPtr,
                         struct CNN_CONFIG_LAYER_CONV* cfgPtr, int inWidth,
                         int inHeight, int inChannel, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;                 // Output matrix size
    int outWidth, outHeight, outChannel;  // Valid convolution output size
    int kRows, kCols;                     // Kernel matrix size

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
    int bRows, bCols;  // Bias matrix size
#endif

#ifdef CNN_WITH_CUDA
    int size;
    int* tmpVec = NULL;
#endif

    // Find output image size
    cnn_run(cnn_config_find_layer_outsize(&outWidth, &outHeight, &outChannel,
                                          inWidth, inHeight, inChannel,
                                          (union CNN_CONFIG_LAYER*)cfgPtr),
            ret, RET);

    // Find allocate size
    outRows = batch;
    outCols = outWidth * outHeight * cfgPtr->filter;

    kRows = cfgPtr->filter;
    kCols = cfgPtr->size * cfgPtr->size * inChannel;

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
    bRows = 1;
    bCols = cfgPtr->filter;

#if defined(CNN_CONV_BIAS_LAYER)
#ifdef DEBUG
#pragma message("cnn_layer_conv_alloc(): Enable convolution layer bias")
#endif
    bCols = outCols;
#endif
#endif

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->kernel, kRows, kCols, 1), ret, ERR);

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
    cnn_run(cnn_mat_alloc(&layerPtr->bias, bRows, bCols, 1), ret, ERR);
#endif

    cnn_run(cnn_mat_alloc(&layerPtr->unroll, outWidth * outHeight * batch,
                          kCols, 1),
            ret, ERR);

#ifdef CNN_WITH_CUDA
    cnn_alloc_cu(layerPtr->indexMap, outWidth * outHeight * kCols, int, ret,
                 ERR);
#else
    cnn_alloc(layerPtr->indexMap, outWidth * outHeight * kCols, int, ret, ERR);
#endif

    // Assing value
    layerPtr->outMat.width = outWidth;
    layerPtr->outMat.height = outHeight;
    layerPtr->inChannel = inChannel;
    layerPtr->outMat.channel = outChannel;

#ifdef CNN_WITH_CUDA
    // Cache index map size
    size = outWidth * outHeight * kCols;

    // Buffer allocation
    cnn_alloc(tmpVec, size, int, ret, ERR);
#endif

    // Initial index mapping
    switch (cfgPtr->pad)
    {
        case CNN_PAD_VALID:
#ifdef CNN_WITH_CUDA
            cnn_conv_unroll_2d_valid(tmpVec, outHeight, outWidth, cfgPtr->size,
                                     inHeight, inWidth, inChannel);

#else
            cnn_conv_unroll_2d_valid(layerPtr->indexMap, outHeight, outWidth,
                                     cfgPtr->size, inHeight, inWidth,
                                     inChannel);
#endif
            break;

        case CNN_PAD_SAME:
#ifdef CNN_WITH_CUDA
            for (int i = 0; i < outWidth * outHeight * kCols; i++)
            {
                tmpVec[i] = -1;
            }

            cnn_conv_unroll_2d_same(tmpVec, outHeight, outWidth, cfgPtr->size,
                                    inHeight, inWidth, inChannel);
#else
            for (int i = 0; i < outWidth * outHeight * kCols; i++)
            {
                layerPtr->indexMap[i] = -1;
            }

            cnn_conv_unroll_2d_same(layerPtr->indexMap, outHeight, outWidth,
                                    cfgPtr->size, inHeight, inWidth, inChannel);
#endif
            break;

        default:
            assert(!"Invalid padding type");
    }

#ifdef CNN_WITH_CUDA
    // Copy memory
    cnn_run_cu(cudaMemcpy(layerPtr->indexMap, tmpVec, size * sizeof(int),
                          cudaMemcpyHostToDevice),
               ret, ERR);
#endif

    goto RET;

ERR:
    cnn_layer_conv_delete(layerPtr);

RET:
#ifdef CNN_WITH_CUDA
    // Free buffer
    cnn_free(tmpVec);
#endif

    return ret;
}

int cnn_layer_bn_alloc(struct CNN_LAYER_BN* layerPtr,
                       struct CNN_CONFIG_LAYER_BN* cfgPtr, int inWidth,
                       int inHeight, int inChannel, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;
    int outWidth, outHeight, outChannel;

#ifdef CNN_WITH_CUDA
    int size;
    float* tmpVec = NULL;
#endif

    // Find output size
    cnn_run(cnn_config_find_layer_outsize(&outWidth, &outHeight, &outChannel,
                                          inWidth, inHeight, inChannel,
                                          (union CNN_CONFIG_LAYER*)cfgPtr),
            ret, RET);

    // Find allocate size
    outRows = batch;
    outCols = inWidth * inHeight * inChannel;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->bnVar, inChannel, 2, 1), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->srcShift, outRows, outCols, 0), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->srcNorm, outRows, outCols, 1), ret, ERR);

    //#ifdef CNN_WITH_CUDA
    //    cnn_alloc_cu(layerPtr->stddev, inChannel, float, ret, ERR);
    //#else
    cnn_alloc(layerPtr->stddev, inChannel * batch, float, ret, ERR);
    //#endif

#ifdef CNN_WITH_CUDA
    cnn_alloc_cu(layerPtr->buf, inWidth * inHeight, float, ret, ERR);
#endif

#ifdef CNN_WITH_CUDA
    // Buffer allocation
    size = inChannel * 2;
    cnn_alloc(tmpVec, size, float, ret, ERR);

    // Set initial gamma, beta
    for (int i = 0; i < inChannel; i++)
    {
        tmpVec[i * 2 + 0] = cfgPtr->rInit;
        tmpVec[i * 2 + 1] = cfgPtr->bInit;
    }

    // Copy memory
    cnn_run_cu(cudaMemcpy(layerPtr->bnVar.mat, tmpVec, size * sizeof(float),
                          cudaMemcpyHostToDevice),
               ret, ERR);

#else
    // Set initial gamma, beta
    for (int i = 0; i < inChannel; i++)
    {
        layerPtr->bnVar.mat[i * 2 + 0] = cfgPtr->rInit;
        layerPtr->bnVar.mat[i * 2 + 1] = cfgPtr->bInit;
    }
#endif

    // Assign value
    layerPtr->outMat.width = outWidth;
    layerPtr->outMat.height = outHeight;
    layerPtr->outMat.channel = outChannel;

    goto RET;

ERR:
    cnn_layer_bn_delete(layerPtr);

RET:
#ifdef CNN_WITH_CUDA
    // Free buffer
    cnn_free(tmpVec);
#endif

    return ret;
}

int cnn_layer_text_alloc(struct CNN_LAYER_TEXT* layerPtr,
                         struct CNN_CONFIG_LAYER_TEXT* cfgPtr, int inWidth,
                         int inHeight, int inChannel, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;                 // Output matrix size
    int outWidth, outHeight, outChannel;  // Valid texture output size
    int wRows, wCols;                     // Weight matrix size;
    int bRows, bCols;                     // Bias matrix size

    const int wSize = 8;

#ifdef CNN_WITH_CUDA
    int nbrSize;
    int* nbrVec = NULL;

    int ctrSize;
    int* ctrVec = NULL;
#endif

    // Find output image size
    cnn_run(cnn_config_find_layer_outsize(&outWidth, &outHeight, &outChannel,
                                          inWidth, inHeight, inChannel,
                                          (union CNN_CONFIG_LAYER*)cfgPtr),
            ret, RET);

    // Find allocate size
    outRows = batch;
    outCols = outWidth * outHeight * outChannel;

    wRows = cfgPtr->filter;
    wCols = wSize * inChannel;

    bRows = 1;
    bCols = cfgPtr->filter;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->weight, wRows, wCols, 1), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->bias, bRows, bCols, 1), ret, ERR);

    cnn_run(cnn_mat_alloc(&layerPtr->nbrUnroll, inWidth * inHeight * inChannel,
                          wSize, 1),
            ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->ctrUnroll, inWidth * inHeight * inChannel,
                          1, 1),
            ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->activ, inWidth * inHeight,
                          inChannel * wSize, 1),
            ret, ERR);

#ifdef CNN_WITH_CUDA
    cnn_alloc_cu(layerPtr->nbrMap, inChannel * inWidth * inHeight * wSize, int,
                 ret, ERR);
    cnn_alloc_cu(layerPtr->ctrMap, inChannel * inWidth * inHeight, int, ret,
                 ERR);
#else
    cnn_alloc(layerPtr->nbrMap, inChannel * inWidth * inHeight * wSize, int,
              ret, ERR);
    cnn_alloc(layerPtr->ctrMap, inChannel * inWidth * inHeight, int, ret, ERR);
#endif

    // Assign value
    layerPtr->outMat.width = outWidth;
    layerPtr->outMat.height = outHeight;
    layerPtr->inChannel = inChannel;
    layerPtr->outMat.channel = outChannel;

#ifdef CNN_WITH_CUDA
    // Cache index map size
    nbrSize = outWidth * outHeight * wCols;
    ctrSize = outWidth * outHeight * inChannel;

    // Buffer allocation
    cnn_alloc(nbrVec, nbrSize, int, ret, ERR);
    cnn_alloc(ctrVec, ctrSize, int, ret, ERR);
#endif

    // Initial index mapping
#ifdef CNN_WITH_CUDA
    cnn_text_unroll(nbrVec, ctrVec, inWidth, inHeight, inChannel);

    // Copy memory
    cnn_run_cu(cudaMemcpy(layerPtr->nbrMap, nbrVec, nbrSize * sizeof(int),
                          cudaMemcpyHostToDevice),
               ret, ERR);
    cnn_run_cu(cudaMemcpy(layerPtr->ctrMap, ctrVec, ctrSize * sizeof(int),
                          cudaMemcpyHostToDevice),
               ret, ERR);
#else
    cnn_text_unroll(layerPtr->nbrMap, layerPtr->ctrMap, inWidth, inHeight,
                    inChannel);
#endif

    goto RET;

ERR:
    cnn_layer_text_delete(layerPtr);

RET:
#ifdef CNN_WITH_CUDA
    // Free buffer
    cnn_free(nbrVec);
    cnn_free(ctrVec);
#endif

    return ret;
}
