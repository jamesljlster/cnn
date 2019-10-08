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

            case CNN_LAYER_TEXT:
                cnn_run(cnn_layer_text_alloc(
                            &cnn->layerList[i].text,
                            (struct CNN_CONFIG_LAYER_TEXT*)&cfg->layerCfg[i],
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
    // Create tensor
    cnn_run_cudnn(cudnnCreateTensorDescriptor(&layerPtr->ten), ret, ERR);
    cnn_run_cudnn(cudnnSetTensor4dDescriptor(
                      layerPtr->ten, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  //
                      batch, inChannel, inHeight, inWidth),
                  ret, ERR);

    // Allocate state space
    cnn_run_cudnn(
        cudnnDropoutGetStatesSize(cnnInit.cudnnHandle, &layerPtr->stateSize),
        ret, ERR);
    cnn_alloc_cu(layerPtr->stateSpace, layerPtr->stateSize, char, ret, ERR);

    // Create dropout descriptor
    cnn_run_cudnn(cudnnCreateDropoutDescriptor(&layerPtr->dropDesc), ret, ERR);
    cnn_run_cudnn(cudnnSetDropoutDescriptor(                      //
                      layerPtr->dropDesc, cnnInit.cudnnHandle,    //
                      cfgPtr->rate,                               //
                      layerPtr->stateSpace, layerPtr->stateSize,  //
                      cnnInit.randSeed),
                  ret, ERR);

    // Allocate reserve space
    cnn_run_cudnn(
        cudnnDropoutGetReserveSpaceSize(layerPtr->ten, &layerPtr->rsvSize), ret,
        ERR);
    cnn_alloc_cu(layerPtr->rsvSpace, layerPtr->rsvSize, char, ret, ERR);

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

#ifdef CNN_WITH_CUDA
    // Create tensor
    cnn_run_cudnn(cudnnCreateTensorDescriptor(&layerPtr->ten), ret, ERR);
    cnn_run_cudnn(cudnnSetTensor4dDescriptor(
                      layerPtr->ten, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  //
                      batch, inChannel, inHeight, inWidth),
                  ret, ERR);
#endif

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

#ifdef CNN_WITH_CUDA
    size_t sizeTmp;
#endif

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

#ifdef CNN_WITH_CUDA
    // Create bias tensor
    cnn_run_cudnn(cudnnCreateTensorDescriptor(&layerPtr->biasTen), ret, ERR);
    cnn_run_cudnn(
        cudnnSetTensor4dDescriptor(layerPtr->biasTen, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,  //
                                   1, outChannel, outHeight, outWidth),
        ret, ERR);

    // Create destination tensor
    cnn_run_cudnn(cudnnCreateTensorDescriptor(&layerPtr->dstTen), ret, ERR);
    cnn_run_cudnn(cudnnSetTensor4dDescriptor(
                      layerPtr->dstTen, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  //
                      batch, outChannel, outHeight, outWidth),
                  ret, ERR);

    // Create reduction descriptor
    cnn_run_cudnn(cudnnCreateReduceTensorDescriptor(&layerPtr->reduDesc), ret,
                  ERR);
    cnn_run_cudnn(cudnnSetReduceTensorDescriptor(
                      layerPtr->reduDesc, CUDNN_REDUCE_TENSOR_ADD,
                      CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
                      CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES),
                  ret, ERR);

    // Find indices size
    cnn_run_cudnn(cudnnGetReductionIndicesSize(
                      cnnInit.cudnnHandle, layerPtr->reduDesc, layerPtr->dstTen,
                      layerPtr->biasTen, &layerPtr->indSize),
                  ret, ERR);

    // Allocate indices space
    cnn_alloc_cu(layerPtr->indData, layerPtr->indSize, char, ret, ERR);

    // Find workspace size
    cnn_run_cudnn(cudnnGetReductionWorkspaceSize(
                      cnnInit.cudnnHandle, layerPtr->reduDesc, layerPtr->dstTen,
                      layerPtr->biasTen, &sizeTmp),
                  ret, ERR);
    cnn_cudnn_ws_size_ext(sizeTmp);

#endif

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

#ifdef CNN_WITH_CUDA
    int outBatch;            // Pooling output batch size
    int vPad = 0, hPad = 0;  // Vertical padding, horizontal padding

    cudnnPoolingMode_t poolMode =
        (cfgPtr->poolType == CNN_POOL_MAX)
            ? CUDNN_POOLING_MAX
            : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

    // Create source tensor
    cnn_run_cudnn(cudnnCreateTensorDescriptor(&layerPtr->srcTen), ret, ERR);
    cnn_run_cudnn(cudnnSetTensor4dDescriptor(
                      layerPtr->srcTen, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  //
                      batch, inChannel, inHeight, inWidth),
                  ret, ERR);

    // Create pooling descriptor
    cnn_run_cudnn(cudnnCreatePoolingDescriptor(&layerPtr->poolDesc), ret, ERR);
    cnn_run_cudnn(
        cudnnSetPooling2dDescriptor(                            //
            layerPtr->poolDesc, poolMode, CUDNN_PROPAGATE_NAN,  //
            cfgPtr->size, cfgPtr->size, vPad, hPad, cfgPtr->size, cfgPtr->size),
        ret, ERR);

    // Create destination tensor
    cnn_run_cudnn(cudnnGetPooling2dForwardOutputDim(
                      layerPtr->poolDesc, layerPtr->srcTen, &outBatch,
                      &outChannel, &outHeight, &outWidth),
                  ret, ERR);
    if (outBatch != batch)
    {
        assert(!"Batch size is not invarriant");
    }

    cnn_run_cudnn(cudnnCreateTensorDescriptor(&layerPtr->dstTen), ret, ERR);
    cnn_run_cudnn(cudnnSetTensor4dDescriptor(
                      layerPtr->dstTen, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  //
                      outBatch, outChannel, outHeight, outWidth),
                  ret, ERR);

#else
    // Find output image size
    cnn_run(cnn_config_find_layer_outsize(&outWidth, &outHeight, &outChannel,
                                          inWidth, inHeight, inChannel,
                                          (union CNN_CONFIG_LAYER*)cfgPtr),
            ret, RET);
#endif

    // Find allocate size
    outRows = batch;
    outCols = outWidth * outHeight * inChannel;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);

#ifndef CNN_WITH_CUDA
    if (cfgPtr->poolType == CNN_POOL_MAX)
    {
        cnn_alloc(layerPtr->indexMat, outRows * outCols, int, ret, ERR);
    }
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
    int outWidth, outHeight, outChannel;  // Convolution output size
    int kRows, kCols;                     // Kernel matrix size

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
    int bRows, bCols;  // Bias matrix size
#endif

#ifdef CNN_WITH_CUDA
    int padSize;   // Padding size
    int outBatch;  // Convolution output batch size
    size_t sizeTmp;

    // Create source tensor
    cnn_run_cudnn(cudnnCreateTensorDescriptor(&layerPtr->srcTen), ret, ERR);
    cnn_run_cudnn(cudnnSetTensor4dDescriptor(
                      layerPtr->srcTen, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  //
                      batch, inChannel, inHeight, inWidth),
                  ret, ERR);

    // Create kernel tensor
    cnn_run_cudnn(cudnnCreateFilterDescriptor(&layerPtr->kernelTen), ret, ERR);
    cnn_run_cudnn(
        cudnnSetFilter4dDescriptor(
            layerPtr->kernelTen, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,  //
            cfgPtr->filter, inChannel, cfgPtr->size, cfgPtr->size),
        ret, ERR);

    // Create bias tensor
#if defined(CNN_CONV_BIAS_FILTER)
    cnn_run_cudnn(cudnnCreateTensorDescriptor(&layerPtr->biasTen), ret, ERR);
    cnn_run_cudnn(
        cudnnSetTensor4dDescriptor(layerPtr->biasTen, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,  //
                                   1, cfgPtr->filter, 1, 1),
        ret, ERR);
#elif defined(CNN_CONV_BIAS_LAYER)
#error Unsupported convolution bias type
#endif

    // Create convolution descriptor
    cnn_run_cudnn(cudnnCreateConvolutionDescriptor(&layerPtr->convDesc), ret,
                  ERR);

    switch (cfgPtr->pad)
    {
        case CNN_PAD_VALID:
            padSize = 0;
            break;

        case CNN_PAD_SAME:
            padSize = cfgPtr->size / 2;
            break;

        default:
            assert(!"Invalid padding type");
    }

    cnn_run_cudnn(cudnnSetConvolution2dDescriptor(
                      layerPtr->convDesc,            //
                      padSize, padSize, 1, 1, 1, 1,  //
                      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                  ret, ERR);

    // Create destination tensor
    cnn_run_cudnn(cudnnGetConvolution2dForwardOutputDim(
                      layerPtr->convDesc, layerPtr->srcTen, layerPtr->kernelTen,
                      &outBatch, &outChannel, &outHeight, &outWidth),
                  ret, ERR);
    if (outBatch != batch)
    {
        assert(!"Batch size is not invarriant");
    }

    cnn_run_cudnn(cudnnCreateTensorDescriptor(&layerPtr->dstTen), ret, ERR);
    cnn_run_cudnn(cudnnSetTensor4dDescriptor(
                      layerPtr->dstTen, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  //
                      outBatch, outChannel, outHeight, outWidth),
                  ret, ERR);

    // Set convolution algorithm
    cnn_run_cudnn(
        cudnnGetConvolutionForwardAlgorithm(
            cnnInit.cudnnHandle,  //
            layerPtr->srcTen, layerPtr->kernelTen, layerPtr->convDesc,
            layerPtr->dstTen,  //
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &layerPtr->convAlgoFW),
        ret, ERR);

    cnn_run_cudnn(cudnnGetConvolutionBackwardFilterAlgorithm(
                      cnnInit.cudnnHandle,  //
                      layerPtr->srcTen, layerPtr->dstTen, layerPtr->convDesc,
                      layerPtr->kernelTen,  //
                      CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
                      &layerPtr->convAlgoBWFilter),
                  ret, ERR);

    cnn_run_cudnn(cudnnGetConvolutionBackwardDataAlgorithm(
                      cnnInit.cudnnHandle,  //
                      layerPtr->kernelTen, layerPtr->dstTen, layerPtr->convDesc,
                      layerPtr->srcTen,  //
                      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
                      &layerPtr->convAlgoBWGrad),
                  ret, ERR);

    // Find workspace size
    cnn_run_cudnn(cudnnGetConvolutionForwardWorkspaceSize(
                      cnnInit.cudnnHandle, layerPtr->srcTen,
                      layerPtr->kernelTen, layerPtr->convDesc, layerPtr->dstTen,
                      layerPtr->convAlgoFW, &sizeTmp),
                  ret, ERR);
    cnn_cudnn_ws_size_ext(sizeTmp);

    cnn_run_cudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                      cnnInit.cudnnHandle, layerPtr->srcTen, layerPtr->dstTen,
                      layerPtr->convDesc, layerPtr->kernelTen,
                      layerPtr->convAlgoBWFilter, &sizeTmp),
                  ret, ERR);
    cnn_cudnn_ws_size_ext(sizeTmp);

    cnn_run_cudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(
                      cnnInit.cudnnHandle, layerPtr->kernelTen,
                      layerPtr->dstTen, layerPtr->convDesc, layerPtr->srcTen,
                      layerPtr->convAlgoBWGrad, &sizeTmp),
                  ret, ERR);
    cnn_cudnn_ws_size_ext(sizeTmp);

#else
    // Find output image size
    cnn_run(cnn_config_find_layer_outsize(&outWidth, &outHeight, &outChannel,
                                          inWidth, inHeight, inChannel,
                                          (union CNN_CONFIG_LAYER*)cfgPtr),
            ret, RET);
#endif

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

#ifndef CNN_WITH_CUDA
    cnn_run(cnn_mat_alloc(&layerPtr->unroll, outWidth * outHeight * batch,
                          kCols, 1),
            ret, ERR);

    cnn_alloc(layerPtr->indexMap, outWidth * outHeight * kCols, int, ret, ERR);
#endif

    // Assing value
    layerPtr->outMat.width = outWidth;
    layerPtr->outMat.height = outHeight;
    layerPtr->inChannel = inChannel;
    layerPtr->outMat.channel = outChannel;

#ifndef CNN_WITH_CUDA
    // Initial index mapping
    switch (cfgPtr->pad)
    {
        case CNN_PAD_VALID:
            cnn_conv_unroll_2d_valid(layerPtr->indexMap, outHeight, outWidth,
                                     cfgPtr->size, inHeight, inWidth,
                                     inChannel);
            break;

        case CNN_PAD_SAME:
            for (int i = 0; i < outWidth * outHeight * kCols; i++)
            {
                layerPtr->indexMap[i] = -1;
            }

            cnn_conv_unroll_2d_same(layerPtr->indexMap, outHeight, outWidth,
                                    cfgPtr->size, inHeight, inWidth, inChannel);
            break;

        default:
            assert(!"Invalid padding type");
    }
#endif

    goto RET;

ERR:
    cnn_layer_conv_delete(layerPtr);

RET:
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
    cnn_run_cudnn(cudnnCreateTensorDescriptor(&layerPtr->srcTen), ret, ERR);
    cnn_run_cudnn(cudnnSetTensor4dDescriptor(                                 //
                      layerPtr->srcTen, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  //
                      batch, inChannel, inHeight, inWidth),
                  ret, ERR);

    cnn_run_cudnn(cudnnCreateTensorDescriptor(&layerPtr->bnTen), ret, ERR);
    cnn_run_cudnn(
        cudnnDeriveBNTensorDescriptor(layerPtr->bnTen, layerPtr->srcTen,
                                      CUDNN_BATCHNORM_SPATIAL),
        ret, ERR);
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

    cnn_run(cnn_mat_alloc(&layerPtr->bnScale, inChannel, 1, 1), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->bnBias, inChannel, 1, 1), ret, ERR);

    cnn_run(cnn_mat_alloc(&layerPtr->saveMean, inChannel, 1, 0), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->saveVar, inChannel, 1, 0), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->runMean, inChannel, 1, 0), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->runVar, inChannel, 1, 0), ret, ERR);

    // Set initial gamma
    for (int i = 0; i < inChannel; i++)
    {
        float tmp = cfgPtr->rInit;

#ifdef CNN_WITH_CUDA
        cnn_run_cu(cudaMemcpy(layerPtr->bnScale.mat + i, &tmp, sizeof(float),
                              cudaMemcpyHostToDevice),
                   ret, ERR);
#else
        layerPtr->bnScale.mat[i] = tmp;
#endif
    }

    // Set initial beta
    for (int i = 0; i < inChannel; i++)
    {
        float tmp = cfgPtr->bInit;

#ifdef CNN_WITH_CUDA
        cnn_run_cu(cudaMemcpy(layerPtr->bnBias.mat + i, &tmp, sizeof(float),
                              cudaMemcpyHostToDevice),
                   ret, ERR);
#else
        layerPtr->bnBias.mat[i] = tmp;
#endif
    }

    // Set initial running variance
    for (int i = 0; i < inChannel; i++)
    {
        float tmp = 1.0;

#ifdef CNN_WITH_CUDA
        cnn_run_cu(cudaMemcpy(layerPtr->runVar.mat + i, &tmp, sizeof(float),
                              cudaMemcpyHostToDevice),
                   ret, ERR);
#else
        layerPtr->runVar.mat[i] = tmp;
#endif
    }

    // Assign value
    layerPtr->outMat.width = outWidth;
    layerPtr->outMat.height = outHeight;
    layerPtr->outMat.channel = outChannel;

    goto RET;

ERR:
    cnn_layer_bn_delete(layerPtr);

RET:

    return ret;
}

int cnn_layer_text_alloc(struct CNN_LAYER_TEXT* layerPtr,
                         struct CNN_CONFIG_LAYER_TEXT* cfgPtr, int inWidth,
                         int inHeight, int inChannel, int batch)
{
    int ret = CNN_NO_ERROR;
    int outRows, outCols;                 // Output matrix size
    int outWidth, outHeight, outChannel;  // Valid texture output size
    int aRows, aCols;                     // Alpha matrix size
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

    aRows = 1;
    aCols = inChannel;

    wRows = cfgPtr->filter;
    wCols = wSize * inChannel;

    bRows = 1;
    bCols = cfgPtr->filter;

    // Allocate memory
    cnn_run(cnn_mat_alloc(&layerPtr->outMat.data, outRows, outCols, 1), ret,
            ERR);

    cnn_run(cnn_mat_alloc(&layerPtr->alpha, aRows, aCols, 1), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->weight, wRows, wCols, 1), ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->bias, bRows, bCols, 1), ret, ERR);

    cnn_run(cnn_mat_alloc(&layerPtr->nbrUnroll,
                          inWidth * inHeight * inChannel * batch, wSize, 1),
            ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->ctrUnroll,
                          inWidth * inHeight * inChannel * batch, 1, 1),
            ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->diff, inWidth * inHeight * batch,
                          inChannel * wSize, 1),
            ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->scale, inWidth * inHeight * batch,
                          inChannel * wSize, 1),
            ret, ERR);
    cnn_run(cnn_mat_alloc(&layerPtr->activ, inWidth * inHeight * batch,
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

    // Initial alpha matrix
    for (int i = 0; i < inChannel; i++)
    {
        layerPtr->alpha.mat[i] = cfgPtr->aInit;
    }

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
