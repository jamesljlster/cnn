#ifndef __CNN_H__
#define __CNN_H__

enum CNN_RETVAL
{
	CNN_NO_ERROR = 0,
	CNN_MEM_FAILED = -1,
	CNN_INVALID_ARG = -3,
	CNN_INVALID_SHAPE = -4
};

enum CNN_LAYER_TYPE
{
	CNN_LAYER_FC = 1,
	CNN_LAYER_AFUNC = 2,
	CNN_LAYER_CONV = 3
};

enum CNN_ACTIVATION_FUNC
{
	CNN_SOFTMAX = 0,
	CNN_RELU = 1,
	CNN_SWISH = 2
};

typedef struct CNN* cnn_t;
typedef struct CNN_CONFIG* cnn_config_t;

#ifdef __cplusplus
extern "C" {
#endif

int cnn_config_create(cnn_config_t* cfgPtr);
int cnn_config_clone(cnn_config_t* dstPtr, const cnn_config_t src);
void cnn_config_delete(cnn_config_t cfg);

int cnn_config_set_input_size(cnn_config_t cfg, int width, int height);
int cnn_config_set_batch_size(cnn_config_t cfg, int batchSize);
int cnn_config_set_layers(cnn_config_t cfg, int layers);
int cnn_config_set_full_connect(cnn_config_t cfg, int layerIndex, int size);
int cnn_config_set_activation(cnn_config_t cfg, int layerIndex, int aFuncID);
int cnn_config_set_convolution(cnn_config_t cfg, int layerIndex, int convDim, int size);

int cnn_create(cnn_t* cnnPtr, const cnn_config_t cfg);
void cnn_delete(cnn_t cnn);

void cnn_forward(cnn_t cnn, float* inputMat, float* outputMat);
void cnn_bp(cnn_t cnn, float lRate, float* errGrad);
int cnn_training(cnn_t cnn, float* inputMat, float* desireMat, float* outputMat, float* errMat);
int cnn_training_custom(cnn_t cnn, float lRate, float* inputMat, float* desireMat, float* outputMat, float* errMat);

void cnn_rand_network(cnn_t cnn);
void cnn_zero_network(cnn_t cnn);

#ifdef __cplusplus
}
#endif

#endif
