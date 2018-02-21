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
	CNN_INFO_NOT_FOUND = -8
};

typedef enum CNN_LAYER_TYPE
{
	CNN_LAYER_INPUT = 0,
	CNN_LAYER_FC = 1,
	CNN_LAYER_AFUNC = 2,
	CNN_LAYER_CONV = 3,
	CNN_LAYER_POOL = 4,
	CNN_LAYER_DROP = 5
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

typedef enum CNN_ACTIVATION_FUNC
{
	CNN_SOFTMAX = 0,
	CNN_RELU = 1,
	CNN_SWISH = 2
} cnn_afunc_t;

typedef struct CNN* cnn_t;
typedef struct CNN_CONFIG* cnn_config_t;

#ifdef __cplusplus
extern "C" {
#endif

int cnn_config_create(cnn_config_t* cfgPtr);
int cnn_config_clone(cnn_config_t* dstPtr, const cnn_config_t src);
void cnn_config_delete(cnn_config_t cfg);

int cnn_config_set_input_size(cnn_config_t cfg, int width, int height, int channel);
void cnn_config_get_input_size(cnn_config_t cfg, int* wPtr, int* hPtr, int* cPtr);

int cnn_config_set_batch_size(cnn_config_t cfg, int batchSize);
void cnn_config_get_batch_size(cnn_config_t cfg, int* batchPtr);

int cnn_config_set_layers(cnn_config_t cfg, int layers);
void cnn_config_get_layers(cnn_config_t cfg, int* layersPtr);
int cnn_config_get_layer_type(cnn_config_t cfg, int layerIndex, cnn_layer_t* typePtr);

int cnn_config_set_full_connect(cnn_config_t cfg, int layerIndex, int size);
int cnn_config_get_full_connect(cnn_config_t cfg, int layerIndex, int* sizePtr);

int cnn_config_set_activation(cnn_config_t cfg, int layerIndex, cnn_afunc_t aFuncID);
int cnn_config_get_activation(cnn_config_t cfg, int layerIndex, cnn_afunc_t* idPtr);

int cnn_config_set_convolution(cnn_config_t cfg, int layerIndex, cnn_dim_t convDim, int size);
int cnn_config_get_convolution(cnn_config_t cfg, int layerIndex, cnn_dim_t* dimPtr,
		int* sizePtr);

int cnn_config_set_pooling(cnn_config_t cfg, int layerIndex, cnn_dim_t dim, cnn_pool_t type,
		int size);
int cnn_config_get_pooling(cnn_config_t cfg, int layerIndex, cnn_dim_t* dimPtr,
		cnn_pool_t* typePtr, int* sizePtr);

int cnn_config_set_dropout(cnn_config_t cfg, int layerIndex, float rate);
int cnn_config_get_dropout(cnn_config_t cfg, int layerIndex, float* ratePtr);

void cnn_config_set_learning_rate(cnn_config_t cfg, float lRate);
void cnn_config_get_learning_rate(cnn_config_t cfg, float* lRatePtr);

int cnn_config_import(cnn_config_t* cfgPtr, const char* fPath);
int cnn_config_export(cnn_config_t cfg, const char* fPath);

cnn_config_t cnn_get_config(cnn_t cnn);

int cnn_create(cnn_t* cnnPtr, const cnn_config_t cfg);
void cnn_delete(cnn_t cnn);

void cnn_forward(cnn_t cnn, float* inputMat, float* outputMat);
void cnn_bp(cnn_t cnn, float lRate, float* errGrad);
int cnn_training(cnn_t cnn, float* inputMat, float* desireMat, float* outputMat, float* errMat);
int cnn_training_custom(cnn_t cnn, float lRate, float* inputMat, float* desireMat, float* outputMat, float* errMat);

void cnn_rand_network(cnn_t cnn);
void cnn_zero_network(cnn_t cnn);

int cnn_import(cnn_t* cnnPtr, const char* fPath);
int cnn_export(cnn_t cnn, const char* fPath);

#ifdef __cplusplus
}
#endif

#endif
