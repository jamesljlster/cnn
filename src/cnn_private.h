#ifndef __CNN_PRIVATE_H__
#define __CNN_PRIVATE_H__

#include "cnn.h"
#include "cnn_macro.h"
#include "cnn_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    // Private config functions
    int cnn_config_init(cnn_config_t cfg);
    void cnn_config_struct_delete(struct CNN_CONFIG* cfg);
    int cnn_config_struct_clone(struct CNN_CONFIG* dstPtr,
                                const struct CNN_CONFIG* src);

    int cnn_config_find_layer_outsize(int* outWPtr, int* outHPtr, int* outCPtr,
                                      int inWidth, int inHeight, int inChannel,
                                      union CNN_CONFIG_LAYER* layerCfg);

    // Private allocate functions
    int cnn_mat_alloc(struct CNN_MAT* matPtr, int rows, int cols, int needGrad);

    int cnn_layer_input_alloc(struct CNN_LAYER_INPUT* layerPtr,
                              struct CNN_CONFIG_LAYER_INPUT* cfgPtr,
                              int inWidth, int inHeight, int inChannel,
                              int batch);
    int cnn_layer_fc_alloc(struct CNN_LAYER_FC* layerPtr,
                           struct CNN_CONFIG_LAYER_FC* cfgPtr, int inWidth,
                           int inHeight, int inChannel, int batch);
    int cnn_layer_activ_alloc(struct CNN_LAYER_ACTIV* layerPtr,
                              struct CNN_CONFIG_LAYER_ACTIV* cfgPtr,
                              int inWidth, int inHeight, int inChannel,
                              int batch);
    int cnn_layer_conv_alloc(struct CNN_LAYER_CONV* layerPtr,
                             struct CNN_CONFIG_LAYER_CONV* cfgPtr, int inWidth,
                             int inHeight, int inChannel, int batch);
    int cnn_layer_pool_alloc(struct CNN_LAYER_POOL* layerPtr,
                             struct CNN_CONFIG_LAYER_POOL* cfgPtr, int inWidth,
                             int inHeight, int inChannel, int batch);
    int cnn_layer_drop_alloc(struct CNN_LAYER_DROP* layerPtr,
                             struct CNN_CONFIG_LAYER_DROP* cfgPtr, int inWidth,
                             int inHeight, int inChannel, int batch);
    int cnn_layer_bn_alloc(struct CNN_LAYER_BN* layerPtr,
                           struct CNN_CONFIG_LAYER_BN* cfgPtr, int inWidth,
                           int inHeight, int inChannel, int batch);
    int cnn_layer_text_alloc(struct CNN_LAYER_TEXT* layerPtr,
                             struct CNN_CONFIG_LAYER_TEXT* cfgPtr, int inWidth,
                             int inHeight, int inChannel, int batch);
    int cnn_layer_rbfact_alloc(struct CNN_LAYER_RBFACT* layerPtr,
                               struct CNN_CONFIG_LAYER_RBFACT* cfgPtr,
                               int inWidth, int inHeight, int inChannel,
                               int batch);

    int cnn_network_alloc(struct CNN* cnn);

    // Private delete functions
    void cnn_mat_delete(struct CNN_MAT* matPtr);

    void cnn_layer_input_delete(struct CNN_LAYER_INPUT* layerPtr);
    void cnn_layer_activ_delete(struct CNN_LAYER_ACTIV* layerPtr);
    void cnn_layer_fc_delete(struct CNN_LAYER_FC* layerPtr);
    void cnn_layer_conv_delete(struct CNN_LAYER_CONV* layerPtr);
    void cnn_layer_pool_delete(struct CNN_LAYER_POOL* layerPtr);
    void cnn_layer_drop_delete(struct CNN_LAYER_DROP* layerPtr);
    void cnn_layer_bn_delete(struct CNN_LAYER_BN* layerPtr);
    void cnn_layer_text_delete(struct CNN_LAYER_TEXT* layerPtr);
    void cnn_layer_rbfact_delete(struct CNN_LAYER_RBFACT* layerPtr);

    void cnn_network_delete(struct CNN* cnn);
    void cnn_struct_delete(struct CNN* cnn);

    // Private clone function
    void cnn_clone_network_detail(struct CNN* dst, const struct CNN* src);

#ifdef __cplusplus
}
#endif

#endif
