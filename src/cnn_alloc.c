#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "cnn.h"
#include "cnn_builtin_math.h"
#include "cnn_calc.h"
#include "cnn_private.h"

int cnn_network_alloc(struct CNN* cnn)
{
    int i;
    int ret = CNN_NO_ERROR;
    int tmpWidth, tmpHeight, tmpChannel;

    struct CNN_CONFIG* cfg;

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
                cnn_run(
                    cnn_layer_input_alloc(&cnn->layerList[i], tmpWidth,
                                          tmpHeight, tmpChannel, cfg->batch),
                    ret, ERR);
                break;

            case CNN_LAYER_FC:
                cnn_run(cnn_layer_fc_alloc(
                            &cnn->layerList[i].fc, tmpWidth, tmpHeight,
                            tmpChannel, cfg->layerCfg[i].fc.size, cfg->batch),
                        ret, ERR);
                break;

            case CNN_LAYER_ACTIV:
                cnn_run(cnn_layer_activ_alloc(
                            &cnn->layerList[i].activ, tmpWidth, tmpHeight,
                            tmpChannel, cfg->batch, cfg->layerCfg[i].activ.id),
                        ret, ERR);
                break;

            case CNN_LAYER_CONV:
                cnn_run(cnn_layer_conv_alloc(
                            &cnn->layerList[i].conv, tmpWidth, tmpHeight,
                            tmpChannel, cfg->layerCfg[i].conv.filter,
                            cfg->layerCfg[i].conv.pad,
                            cfg->layerCfg[i].conv.size, cfg->batch),
                        ret, ERR);
                break;

            case CNN_LAYER_POOL:
                cnn_run(cnn_layer_pool_alloc(
                            &cnn->layerList[i].pool, tmpWidth, tmpHeight,
                            tmpChannel, cfg->layerCfg[i].pool.size, cfg->batch),
                        ret, ERR);
                break;

            case CNN_LAYER_DROP:
                cnn_run(cnn_layer_drop_alloc(&cnn->layerList[i].drop, tmpWidth,
                                             tmpHeight, tmpChannel, cfg->batch),
                        ret, ERR);
                break;

            case CNN_LAYER_BN:
                cnn_run(cnn_layer_bn_alloc(&cnn->layerList[i].bn, tmpWidth,
                                           tmpHeight, tmpChannel, cfg->batch,
                                           cfg->layerCfg[i].bn.rInit,
                                           cfg->layerCfg[i].bn.bInit),
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
    cnn_alloc(matPtr->mat, rows * cols, float, ret, ERR);
    if (needGrad > 0)
    {
        cnn_alloc(matPtr->grad, rows * cols, float, ret, ERR);
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

int cnn_layer_input_alloc(union CNN_LAYER* layerPtr, int inWidth, int inHeight,
                          int inChannel, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;

    // Find allocate size
    outRows = batch;
    outCols = inWidth * inHeight * inChannel;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 0), ret,
            ERR);

    // Assign value
    layerPtr->outMat.width = inWidth;
    layerPtr->outMat.height = inHeight;
    layerPtr->outMat.channel = inChannel;

    goto RET;

ERR:
    cnn_layer_input_delete(layerPtr);

RET:
    return ret;
}

int cnn_layer_drop_alloc(struct CNN_LAYER_DROP* layerPtr, int inWidth,
                         int inHeight, int inChannel, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;

    // Find allocate size
    outRows = batch;
    outCols = inWidth * inHeight * inChannel;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);
    cnn_alloc(layerPtr->mask, outRows * outCols, int, ret, ERR);

    // Assign value
    layerPtr->outMat.width = inWidth;
    layerPtr->outMat.height = inHeight;
    layerPtr->outMat.channel = inChannel;

    goto RET;

ERR:
    cnn_layer_drop_delete(layerPtr);

RET:
    return ret;
}

int cnn_layer_activ_alloc(struct CNN_LAYER_ACTIV* layerPtr, int inWidth,
                          int inHeight, int inChannel, int batch, int activID)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;
    int gradRows, gradCols;

    // Find allocate size
    outRows = batch;
    outCols = inWidth * inHeight * inChannel;

    gradRows = batch;
    gradCols = outCols;

    if (activID == CNN_SOFTMAX)
    {
        gradRows = batch * outCols;
    }

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->gradMat, gradRows, gradCols, 0), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->buf, outRows, outCols, 0), ret, ERR);

    // Assign value
    layerPtr->outMat.width = inWidth;
    layerPtr->outMat.height = inHeight;
    layerPtr->outMat.channel = inChannel;

    goto RET;

ERR:
    cnn_layer_activ_delete(layerPtr);

RET:
    return ret;
}

int cnn_layer_fc_alloc(struct CNN_LAYER_FC* layerPtr, int inWidth, int inHeight,
                       int inChannel, int outSize, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;  // Output matrix size
    int wRows, wCols;      // Weight matrix size
    int bRows, bCols;      // Bias matrix size

    // Find allocate size
    outRows = batch;
    outCols = outSize;

    wRows = inWidth * inHeight * inChannel;
    wCols = outSize;

    bRows = 1;
    bCols = outSize;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->weight, wRows, wCols, 1), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->bias, bRows, bCols, 1), ret, ERR);

    // Assign value
    layerPtr->outMat.width = outCols;
    layerPtr->outMat.height = 1;
    layerPtr->outMat.channel = 1;

    goto RET;

ERR:
    cnn_layer_fc_delete(layerPtr);

RET:
    return ret;
}

int cnn_layer_pool_alloc(struct CNN_LAYER_POOL* layerPtr, int inWidth,
                         int inHeight, int inChannel, int size, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;     // Output matrix size
    int outWidth, outHeight;  // Valid pooling output size

    // Find output image size
    outWidth = inWidth / size;
    outHeight = inHeight / size;

    // Checking
    if (outWidth <= 0 || outHeight <= 0)
    {
        ret = CNN_INVALID_SHAPE;
        goto RET;
    }

    // Find allocate size
    outRows = batch;
    outCols = outWidth * outHeight * inChannel;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);
    cnn_alloc(layerPtr->indexMat, outRows * outCols, int, ret, ERR);

    // Assing value
    layerPtr->outMat.width = outWidth;
    layerPtr->outMat.height = outHeight;
    layerPtr->outMat.channel = inChannel;

    goto RET;

ERR:
    cnn_layer_pool_delete(layerPtr);

RET:
    return ret;
}

int cnn_layer_conv_alloc(struct CNN_LAYER_CONV* layerPtr, int inWidth,
                         int inHeight, int inChannel, int filter, int padding,
                         int size, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;     // Output matrix size
    int outWidth, outHeight;  // Valid convolution output size
    int kRows, kCols;         // Kernel matrix size

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
    int bRows, bCols;  // Bias matrix size
#endif

    // Find output image size
    switch (padding)
    {
        case CNN_PAD_VALID:
            outWidth = inWidth - size + 1;
            outHeight = inHeight - size + 1;
            break;

        case CNN_PAD_SAME:
            outWidth = inWidth;
            outHeight = inHeight;
            break;

        default:
            assert(!"Invalid padding type");
    }

    // Checking
    if (outWidth <= 0 || outHeight <= 0)
    {
        ret = CNN_INVALID_SHAPE;
        goto RET;
    }

    // Find allocate size
    outRows = batch;
    outCols = outWidth * outHeight * filter;

    kRows = filter;
    kCols = size * size * inChannel;

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
    bRows = 1;
    bCols = filter;

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

    cnn_alloc(layerPtr->indexMap, outWidth * outHeight * kCols, int, ret, ERR);

    // Assing value
    layerPtr->outMat.width = outWidth;
    layerPtr->outMat.height = outHeight;
    layerPtr->inChannel = inChannel;
    layerPtr->outMat.channel = filter;

    // Initial index mapping
    switch (padding)
    {
        case CNN_PAD_VALID:
            cnn_conv_unroll_2d_valid(layerPtr->indexMap, outHeight, outWidth,
                                     size, inHeight, inWidth, inChannel);
            break;

        case CNN_PAD_SAME:
            for (int i = 0; i < outWidth * outHeight * kCols; i++)
            {
                layerPtr->indexMap[i] = -1;
            }

            cnn_conv_unroll_2d_same(layerPtr->indexMap, outHeight, outWidth,
                                    size, inHeight, inWidth, inChannel);
            break;

        default:
            assert(!"Invalid padding type");
    }

    goto RET;

ERR:
    cnn_layer_conv_delete(layerPtr);

RET:
    return ret;
}

int cnn_layer_bn_alloc(struct CNN_LAYER_BN* layerPtr, int inWidth, int inHeight,
                       int inChannel, int batch, float rInit, float bInit)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;

    // Find allocate size
    outRows = batch;
    outCols = inWidth * inHeight * inChannel;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->bnVar, inChannel, 2, 1), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->srcShift, outRows, outCols, 0), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->srcNorm, outRows, outCols, 1), ret, ERR);
    cnn_alloc(layerPtr->stddev, inChannel, float, ret, ERR);

    // Set initial gamma, beta
    for (int i = 0; i < inChannel; i++)
    {
        layerPtr->bnVar.mat[i * 2 + 0] = rInit;
        layerPtr->bnVar.mat[i * 2 + 1] = bInit;
    }

    // Assign value
    layerPtr->outMat.width = inWidth;
    layerPtr->outMat.height = inHeight;
    layerPtr->outMat.channel = inChannel;

    goto RET;

ERR:
    cnn_layer_bn_delete(layerPtr);

RET:
    return ret;
}
