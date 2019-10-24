#include <stdlib.h>
#include <string.h>

#include "cnn.h"
#include "cnn_private.h"

void cnn_delete(cnn_t cnn)
{
    if (cnn != NULL)
    {
        cnn_struct_delete(cnn);
        cnn_free(cnn);
    }
}

void cnn_struct_delete(struct CNN* cnn)
{
    // Delete network
    cnn_network_delete(cnn);

    // Delete config
    cnn_config_struct_delete(&cnn->cfg);

    // Zero memroy
    memset(cnn, 0, sizeof(struct CNN));
}

void cnn_mat_delete(struct CNN_MAT* matPtr)
{
    // Free memory
#ifdef CNN_WITH_CUDA
    cnn_free_cu(matPtr->mat);
    cnn_free_cu(matPtr->grad);
#else
    cnn_free(matPtr->mat);
    cnn_free(matPtr->grad);
#endif

    // Zero memory
    memset(matPtr, 0, sizeof(struct CNN_MAT));
}

void cnn_layer_input_delete(struct CNN_LAYER_INPUT* layerPtr)
{
    // Free memory
    cnn_mat_delete(&layerPtr->outMat.data);

    // Zero memory
    memset(layerPtr, 0, sizeof(struct CNN_LAYER_INPUT));
}

void cnn_layer_drop_delete(struct CNN_LAYER_DROP* layerPtr)
{
    // Free memory
    cnn_mat_delete(&layerPtr->outMat.data);
    cnn_free(layerPtr->mask);

#ifdef CNN_WITH_CUDA
    // Destroy tensor
    cudnnDestroyTensorDescriptor(layerPtr->ten);

    // Destroy dropout descriptor
    cudnnDestroyDropoutDescriptor(layerPtr->dropDesc);

    // Free state space
    cnn_free_cu(layerPtr->stateSpace);

    // Free reserve space
    cnn_free_cu(layerPtr->rsvSpace);
#endif

    // Zero memory
    memset(layerPtr, 0, sizeof(struct CNN_LAYER_DROP));
}

void cnn_layer_activ_delete(struct CNN_LAYER_ACTIV* layerPtr)
{
    // Free memory
    cnn_mat_delete(&layerPtr->outMat.data);
    cnn_mat_delete(&layerPtr->gradMat);
    cnn_mat_delete(&layerPtr->buf);

#ifdef CNN_WITH_CUDA
    // Destroy tensor
    cudnnDestroyTensorDescriptor(layerPtr->ten);
#endif

    // Zero memory
    memset(layerPtr, 0, sizeof(struct CNN_LAYER_ACTIV));
}

void cnn_layer_fc_delete(struct CNN_LAYER_FC* layerPtr)
{
    // Free memory
    cnn_mat_delete(&layerPtr->outMat.data);
    cnn_mat_delete(&layerPtr->weight);
    cnn_mat_delete(&layerPtr->bias);

#ifdef CNN_WITH_CUDA
    // Destroy tensor
    cudnnDestroyTensorDescriptor(layerPtr->biasTen);
    cudnnDestroyTensorDescriptor(layerPtr->dstTen);

    // Destroy reduction descriptor
    cudnnDestroyReduceTensorDescriptor(layerPtr->reduDesc);

    // Free indices space
    cnn_free_cu(layerPtr->indData);
#endif

    // Zero memory
    memset(layerPtr, 0, sizeof(struct CNN_LAYER_FC));
}

void cnn_layer_conv_delete(struct CNN_LAYER_CONV* layerPtr)
{
    // Free memory
    cnn_mat_delete(&layerPtr->outMat.data);
    cnn_mat_delete(&layerPtr->kernel);

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
    cnn_mat_delete(&layerPtr->bias);
#endif

#ifdef CNN_WITH_CUDA
    // Destroy convolution descriptor
    cudnnDestroyConvolutionDescriptor(layerPtr->convDesc);

    // Destroy tensor
    cudnnDestroyTensorDescriptor(layerPtr->srcTen);
    cudnnDestroyTensorDescriptor(layerPtr->dstTen);
    cudnnDestroyFilterDescriptor(layerPtr->kernelTen);

#if defined(CNN_CONV_BIAS_FILTER)
    cudnnDestroyTensorDescriptor(layerPtr->biasTen);
#elif defined(CNN_CONV_BIAS_LAYER)
#error Unsupported convolution bias type
#endif

    // cnn_free_cu(layerPtr->indexMap);
#else
    cnn_mat_delete(&layerPtr->unroll);
    cnn_free(layerPtr->indexMap);
#endif

    // Zero memory
    memset(layerPtr, 0, sizeof(struct CNN_LAYER_CONV));
}

void cnn_layer_pool_delete(struct CNN_LAYER_POOL* layerPtr)
{
    // Free memory
    cnn_mat_delete(&layerPtr->outMat.data);

#ifdef CNN_WITH_CUDA
    // Destroy pooling descriptor
    cudnnDestroyPoolingDescriptor(layerPtr->poolDesc);

    // Destroy tensor
    cudnnDestroyTensorDescriptor(layerPtr->srcTen);
    cudnnDestroyTensorDescriptor(layerPtr->dstTen);
#else
    cnn_free(layerPtr->indexMat);
#endif

    // Zero memory
    memset(layerPtr, 0, sizeof(struct CNN_LAYER_POOL));
}

void cnn_layer_bn_delete(struct CNN_LAYER_BN* layerPtr)
{
    // Free memory
    cnn_mat_delete(&layerPtr->outMat.data);

    cnn_mat_delete(&layerPtr->bnScale);
    cnn_mat_delete(&layerPtr->bnBias);

    cnn_mat_delete(&layerPtr->saveMean);
    cnn_mat_delete(&layerPtr->saveVar);
    cnn_mat_delete(&layerPtr->runMean);
    cnn_mat_delete(&layerPtr->runVar);

#ifdef CNN_WITH_CUDA
    // Destroy tensor
    cudnnDestroyTensorDescriptor(layerPtr->srcTen);
    cudnnDestroyTensorDescriptor(layerPtr->bnTen);
#endif

    // Zero memory
    memset(layerPtr, 0, sizeof(struct CNN_LAYER_BN));
}

void cnn_layer_text_delete(struct CNN_LAYER_TEXT* layerPtr)
{
    // Free memory
    cnn_mat_delete(&layerPtr->outMat.data);
    cnn_mat_delete(&layerPtr->weight);
    cnn_mat_delete(&layerPtr->bias);
    cnn_mat_delete(&layerPtr->alpha);

    cnn_mat_delete(&layerPtr->nbrUnroll);
    cnn_mat_delete(&layerPtr->ctrUnroll);
    cnn_mat_delete(&layerPtr->diff);
    cnn_mat_delete(&layerPtr->scale);
    cnn_mat_delete(&layerPtr->activ);

#ifdef CNN_WITH_CUDA
    cnn_free_cu(layerPtr->nbrMap);
    cnn_free_cu(layerPtr->ctrMap);
#else
    cnn_free(layerPtr->nbrMap);
    cnn_free(layerPtr->ctrMap);
#endif

    // Zero memory
    memset(layerPtr, 0, sizeof(struct CNN_LAYER_TEXT));
}

void cnn_layer_rbfact_delete(struct CNN_LAYER_RBFACT* layerPtr)
{
    // Free memory
    cnn_mat_delete(&layerPtr->outMat.data);
    cnn_mat_delete(&layerPtr->center);
    cnn_mat_delete(&layerPtr->runVar);
    cnn_mat_delete(&layerPtr->saveVar);

    cnn_free(layerPtr->ws);

    // Zero memory
    memset(layerPtr, 0, sizeof(struct CNN_LAYER_TEXT));
}

void cnn_network_delete(struct CNN* cnn)
{
    int i;

    if (cnn->layerList != NULL)
    {
        // Delete CNN layers
        for (i = 0; i < cnn->cfg.layers; i++)
        {
            switch (cnn->cfg.layerCfg[i].type)
            {
                case CNN_LAYER_INPUT:
                    cnn_layer_input_delete(&cnn->layerList[i].input);
                    break;

                case CNN_LAYER_FC:
                    cnn_layer_fc_delete(&cnn->layerList[i].fc);
                    break;

                case CNN_LAYER_ACTIV:
                    cnn_layer_activ_delete(&cnn->layerList[i].activ);
                    break;

                case CNN_LAYER_CONV:
                    cnn_layer_conv_delete(&cnn->layerList[i].conv);
                    break;

                case CNN_LAYER_POOL:
                    cnn_layer_pool_delete(&cnn->layerList[i].pool);
                    break;

                case CNN_LAYER_DROP:
                    cnn_layer_drop_delete(&cnn->layerList[i].drop);
                    break;

                case CNN_LAYER_BN:
                    cnn_layer_bn_delete(&cnn->layerList[i].bn);
                    break;

                case CNN_LAYER_TEXT:
                    cnn_layer_text_delete(&cnn->layerList[i].text);
                    break;
            }
        }

        // Free memory
        cnn_free(cnn->layerList);
    }

    cnn->layerList = NULL;
}
