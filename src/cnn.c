#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "cnn.h"
#include "cnn_private.h"

void cnn_mat_clone(struct CNN_MAT* dstPtr, struct CNN_MAT* srcPtr)
{
    // Checking
    assert(dstPtr->rows == srcPtr->rows && dstPtr->cols == srcPtr->cols);

#ifdef CNN_WITH_CUDA
    cudaMemcpy
#else
    memcpy
#endif
        (dstPtr->mat, srcPtr->mat, sizeof(float) * srcPtr->rows * srcPtr->cols);
}

void cnn_get_input_size(cnn_t cnn, int* wPtr, int* hPtr, int* cPtr)
{
    // Assign value
    if (wPtr != NULL)
    {
        *wPtr = cnn->cfg.width;
    }

    if (hPtr != NULL)
    {
        *hPtr = cnn->cfg.height;
    }

    if (cPtr != NULL)
    {
        *cPtr = cnn->cfg.channel;
    }
}

void cnn_get_output_size(cnn_t cnn, int* wPtr, int* hPtr, int* cPtr)
{
    int index = cnn->cfg.layers - 1;

    // Assign value
    if (wPtr != NULL)
    {
        *wPtr = cnn->layerList[index].outMat.width;
    }

    if (hPtr != NULL)
    {
        *hPtr = cnn->layerList[index].outMat.height;
    }

    if (cPtr != NULL)
    {
        *cPtr = cnn->layerList[index].outMat.channel;
    }
}

int cnn_resize_batch(cnn_t* cnnPtr, int batchSize)
{
    int ret = CNN_NO_ERROR;
    cnn_config_t tmpCfg = NULL;
    cnn_t tmpCnn = NULL;

    // Clone config
    cnn_run(cnn_config_clone(&tmpCfg, cnn_get_config(*cnnPtr)), ret, RET);

    // Set new batch size
    cnn_run(cnn_config_set_batch_size(tmpCfg, batchSize), ret, RET);

    // Create new cnn
    cnn_run(cnn_create(&tmpCnn, tmpCfg), ret, RET);

    // Clone network detail
    cnn_clone_network_detail(tmpCnn, *cnnPtr);

    // Replace cnn
    cnn_delete(*cnnPtr);
    *cnnPtr = tmpCnn;
    goto RET;

RET:
    cnn_config_delete(tmpCfg);
    return ret;
}

int cnn_clone(cnn_t* dstPtr, const cnn_t src)
{
    int ret = CNN_NO_ERROR;
    cnn_t tmpCnn = NULL;

    // Create cnn
    cnn_run(cnn_create(&tmpCnn, &src->cfg), ret, RET);

    // Clone network detail
    cnn_clone_network_detail(tmpCnn, src);

    // Assign value
    *dstPtr = tmpCnn;
    goto RET;

RET:
    return ret;
}

void cnn_clone_network_detail(struct CNN* dst, const struct CNN* src)
{
    int i;

    union CNN_LAYER* dstLayerList;
    union CNN_LAYER* srcLayerList;
    const struct CNN_CONFIG* cfgRef;

    // Set reference
    dstLayerList = dst->layerList;
    srcLayerList = src->layerList;
    cfgRef = &src->cfg;

#define __cnn_mat_clone(mat) \
    cnn_mat_clone(&dstLayerList[i].mat, &srcLayerList[i].mat)

    // Clone weight and bias
    for (i = 1; i < cfgRef->layers; i++)
    {
        switch (cfgRef->layerCfg[i].type)
        {
            case CNN_LAYER_FC:
                // Clone weight
                __cnn_mat_clone(fc.weight);

                // Clone bias
                __cnn_mat_clone(fc.bias);

                break;

            case CNN_LAYER_CONV:
                // Clone kernel
                __cnn_mat_clone(conv.kernel);

                // Clone bias
#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
                __cnn_mat_clone(conv.bias);
#endif

                break;

            case CNN_LAYER_BN:
                // Clone parameter
                __cnn_mat_clone(bn.bnVar);

                break;

            case CNN_LAYER_TEXT:
                // Clone weight
                __cnn_mat_clone(text.weight);

                // Clone bias
                __cnn_mat_clone(text.bias);

                break;

            case CNN_LAYER_INPUT:
            case CNN_LAYER_ACTIV:
            case CNN_LAYER_POOL:
            case CNN_LAYER_DROP:
                break;
        }
    }
}
