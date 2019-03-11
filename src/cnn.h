/**
 * @author	Zheng-Ling Lai <jamesljlster@gmail.com>
 * @file	cnn.h
 */

#ifndef __CNN_H__
#define __CNN_H__

enum CNN_RETVAL
{
    CNN_NO_ERROR = 0,
    CNN_MEM_FAILED = -1,
    CNN_INVALID_ARG = -3,
    CNN_INVALID_SHAPE = -4,
    CNN_FILE_OP_FAILED = -5,
    CNN_PARSE_FAILED = -6,
    CNN_INVALID_FILE = -7,
    CNN_INFO_NOT_FOUND = -8,
    CNN_CONFLICT = -9,
    CNN_CUDA_RUNTIME_ERROR = -10,
    CNN_NOT_INITIALIZED = -11
};

typedef enum CNN_LAYER_TYPE
{
    CNN_LAYER_INPUT = 0,
    CNN_LAYER_FC = 1,
    CNN_LAYER_ACTIV = 2,
    CNN_LAYER_CONV = 3,
    CNN_LAYER_POOL = 4,
    CNN_LAYER_DROP = 5,
    CNN_LAYER_BN = 6,
    CNN_LAYER_TEXT = 7
} cnn_layer_t;

typedef enum CNN_POOL_TYPE
{
    CNN_POOL_MAX = 0,
    CNN_POOL_AVG = 1
} cnn_pool_t;

typedef enum CNN_DIM_TYPE
{
    CNN_DIM_1D = 1,
    CNN_DIM_2D = 2
} cnn_dim_t;

typedef enum CNN_PAD_TYPE
{
    CNN_PAD_VALID = 0,
    CNN_PAD_SAME = 1
} cnn_pad_t;

typedef enum CNN_ACTIV_TYPE
{
    CNN_SOFTMAX = 0,
    CNN_RELU = 1,
    CNN_SWISH = 2,
    CNN_SIGMOID = 3,
    CNN_HYPERBOLIC_TANGENT = 4,
    CNN_GAUSSIAN = 5,
    CNN_BENT_IDENTITY = 6,
    CNN_SOFTPLUS = 7,
    CNN_SOFTSIGN = 8,
    CNN_SINC = 9,
    CNN_SINUSOID = 10,
    CNN_IDENTITY = 11
} cnn_activ_t;

typedef struct CNN* cnn_t;
typedef struct CNN_CONFIG* cnn_config_t;

#ifdef __cplusplus
extern "C"
{
#endif

    int cnn_init();
    void cnn_deinit();

    int cnn_config_create(cnn_config_t* cfgPtr);
    int cnn_config_clone(cnn_config_t* dstPtr, const cnn_config_t src);
    void cnn_config_delete(cnn_config_t cfg);

    int cnn_config_set_input_size(cnn_config_t cfg, int width, int height,
                                  int channel);
    void cnn_config_get_input_size(cnn_config_t cfg, int* wPtr, int* hPtr,
                                   int* cPtr);
    void cnn_config_get_output_size(cnn_config_t cfg, int* wPtr, int* hPtr,
                                    int* cPtr);

    int cnn_config_set_batch_size(cnn_config_t cfg, int batchSize);
    void cnn_config_get_batch_size(cnn_config_t cfg, int* batchPtr);

    int cnn_config_set_layers(cnn_config_t cfg, int layers);
    void cnn_config_get_layers(cnn_config_t cfg, int* layersPtr);
    int cnn_config_get_layer_type(cnn_config_t cfg, int layerIndex,
                                  cnn_layer_t* typePtr);

    int cnn_config_append_full_connect(cnn_config_t cfg, int size);
    int cnn_config_set_full_connect(cnn_config_t cfg, int layerIndex, int size);
    int cnn_config_get_full_connect(cnn_config_t cfg, int layerIndex,
                                    int* sizePtr);

    int cnn_config_append_activation(cnn_config_t cfg, cnn_activ_t activID);
    int cnn_config_set_activation(cnn_config_t cfg, int layerIndex,
                                  cnn_activ_t activID);
    int cnn_config_get_activation(cnn_config_t cfg, int layerIndex,
                                  cnn_activ_t* idPtr);

    int cnn_config_append_convolution(cnn_config_t cfg, cnn_pad_t padding,
                                      cnn_dim_t convDim, int filter, int size);
    int cnn_config_set_convolution(cnn_config_t cfg, int layerIndex,
                                   cnn_pad_t padding, cnn_dim_t convDim,
                                   int filter, int size);
    int cnn_config_get_convolution(cnn_config_t cfg, int layerIndex,
                                   cnn_pad_t* padPtr, cnn_dim_t* dimPtr,
                                   int* filterPtr, int* sizePtr);

    int cnn_config_append_pooling(cnn_config_t cfg, cnn_dim_t dim,
                                  cnn_pool_t type, int size);
    int cnn_config_set_pooling(cnn_config_t cfg, int layerIndex, cnn_dim_t dim,
                               cnn_pool_t type, int size);
    int cnn_config_get_pooling(cnn_config_t cfg, int layerIndex,
                               cnn_dim_t* dimPtr, cnn_pool_t* typePtr,
                               int* sizePtr);

    int cnn_config_append_dropout(cnn_config_t cfg, float rate);
    int cnn_config_set_dropout(cnn_config_t cfg, int layerIndex, float rate);
    int cnn_config_get_dropout(cnn_config_t cfg, int layerIndex,
                               float* ratePtr);

    int cnn_config_append_batchnorm(cnn_config_t cfg, float rInit, float bInit);
    int cnn_config_set_batchnorm(cnn_config_t cfg, int layerIndex, float rInit,
                                 float bInit);
    int cnn_config_get_batchnorm(cnn_config_t cfg, int layerIndex,
                                 float* rInitPtr, float* bInitPtr);

    int cnn_config_append_texture(cnn_config_t cfg, cnn_activ_t activID,
                                  int filter, float aInit);
    int cnn_config_set_texture(cnn_config_t cfg, int layerIndex,
                               cnn_activ_t activID, int filter, float aInit);
    int cnn_config_get_texture(cnn_config_t cfg, int layerIndex,
                               cnn_activ_t* idPtr, int* filterPtr,
                               float* aInitPtr);

    void cnn_config_set_learning_rate(cnn_config_t cfg, float lRate);
    void cnn_config_get_learning_rate(cnn_config_t cfg, float* lRatePtr);

    int cnn_config_import(cnn_config_t* cfgPtr, const char* fPath);
    int cnn_config_export(cnn_config_t cfg, const char* fPath);

    int cnn_config_compare(const cnn_config_t src1, const cnn_config_t src2);

    cnn_config_t cnn_get_config(cnn_t cnn);

    void cnn_get_input_size(cnn_t cnn, int* wPtr, int* hPtr, int* cPtr);
    void cnn_get_output_size(cnn_t cnn, int* wPtr, int* hPtr, int* cPtr);

    int cnn_create(cnn_t* cnnPtr, const cnn_config_t cfg);
    int cnn_clone(cnn_t* dstPtr, const cnn_t src);
    void cnn_delete(cnn_t cnn);

    void cnn_set_dropout_enabled(cnn_t cnn, int enable);
    int cnn_resize_batch(cnn_t* cnnPtr, int batchSize);

    void cnn_forward(cnn_t cnn, float* inputMat, float* outputMat);
    void cnn_backward(cnn_t cnn, float* errGrad);
    void cnn_update(cnn_t cnn, float lRate, float gradLimit);
    int cnn_training(cnn_t cnn, float* inputMat, float* desireMat,
                     float* outputMat, float* errMat, float gradLimit);
    int cnn_training_custom(cnn_t cnn, float lRate, float* inputMat,
                            float* desireMat, float* outputMat, float* errMat,
                            float gradLimit);

    void cnn_rand_network(cnn_t cnn);
    void cnn_zero_network(cnn_t cnn);

    int cnn_import(cnn_t* cnnPtr, const char* fPath);
    int cnn_export(cnn_t cnn, const char* fPath);

#ifdef __cplusplus
}
#endif

#endif
