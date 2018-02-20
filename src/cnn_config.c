#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn.h"
#include "cnn_private.h"
#include "cnn_builtin_math.h"

#define CNN_DEFAULT_INPUT_WIDTH 32
#define CNN_DEFAULT_INPUT_HEIGHT 32
#define CNN_DEFAULT_INPUT_CHANNEL 3
#define CNN_DEFAULT_BATCH 1
#define CNN_DEFAULT_LAYERS 1
#define CNN_DEFAULT_FC_SIZE 16
#define CNN_DEFAULT_LRATE 0.001

int cnn_config_clone(cnn_config_t* cfgPtr, const cnn_config_t src)
{
	int ret = CNN_NO_ERROR;
	struct CNN_CONFIG* tmpCfg;

	// Memory allocation
	cnn_alloc(tmpCfg, 1, struct CNN_CONFIG, ret, RET);

	// Clone config struct
	cnn_run(cnn_config_struct_clone(tmpCfg, src), ret, ERR);

	// Assing value
	*cfgPtr = tmpCfg;

	goto RET;

ERR:
	cnn_config_delete(tmpCfg);

RET:
	return ret;
}

int cnn_config_struct_clone(struct CNN_CONFIG* dst, const struct CNN_CONFIG* src)
{
	int ret = CNN_NO_ERROR;

	// Copy setting
	memcpy(dst, src, sizeof(struct CNN_CONFIG));

	// Memory allocation
	cnn_alloc(dst->layerCfg, dst->layers, union CNN_CONFIG_LAYER, ret, ERR);

	// Copy layer setting
	memcpy(dst->layerCfg, src->layerCfg, dst->layers * sizeof(union CNN_CONFIG_LAYER));

	goto RET;

ERR:
	cnn_config_struct_delete(dst);

RET:
	return ret;
}

int cnn_config_create(cnn_config_t* cfgPtr)
{
	int ret = CNN_NO_ERROR;
	struct CNN_CONFIG* tmpCfg;

	// Memory allocation
	cnn_alloc(tmpCfg, 1, struct CNN_CONFIG, ret, RET);

	// Initial config
	cnn_run(cnn_config_init(tmpCfg), ret, ERR);

	// Assign value
	*cfgPtr = tmpCfg;

	goto RET;

ERR:
	cnn_config_delete(tmpCfg);

RET:
	return ret;
}

int cnn_config_init(cnn_config_t cfg)
{
	int ret = CNN_NO_ERROR;

	// Set default cnn settings
	// Set input size
	cnn_run(cnn_config_set_input_size(cfg, CNN_DEFAULT_INPUT_WIDTH, CNN_DEFAULT_INPUT_HEIGHT,
				CNN_DEFAULT_INPUT_CHANNEL),
			ret, RET);

	// Set batch
	cnn_run(cnn_config_set_batch_size(cfg, CNN_DEFAULT_BATCH),
			ret, RET);

	// Set total layers
	cnn_run(cnn_config_set_layers(cfg, CNN_DEFAULT_LAYERS),
			ret, RET);

RET:
	return ret;
}

void cnn_config_struct_delete(struct CNN_CONFIG* cfg)
{
	cnn_free(cfg->layerCfg);
	memset(cfg, 0, sizeof(struct CNN_CONFIG));
}

void cnn_config_delete(cnn_config_t cfg)
{
	if(cfg != NULL)
	{
		cnn_config_struct_delete(cfg);
		cnn_free(cfg);
	}
}

void cnn_config_set_learning_rate(cnn_config_t cfg, float lRate)
{
	cfg->lRate = lRate;
}

void cnn_config_get_learning_rate(cnn_config_t cfg, float* lRatePtr)
{
	if(lRatePtr != NULL)
	{
		*lRatePtr = cfg->lRate;
	}
}

int cnn_config_set_input_size(cnn_config_t cfg, int width, int height, int channel)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(width <= 0 || height <= 0)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	// Assign value
	cfg->width = width;
	cfg->height = height;
	cfg->channel = channel;

RET:
	return ret;
}

void cnn_config_get_input_size(cnn_config_t cfg, int* wPtr, int* hPtr, int* cPtr)
{
	if(wPtr != NULL)
	{
		*wPtr = cfg->width;
	}

	if(hPtr != NULL)
	{
		*hPtr = cfg->height;
	}

	if(cPtr != NULL)
	{
		*cPtr = cfg->channel;
	}
}

int cnn_config_set_batch_size(cnn_config_t cfg, int batchSize)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(batchSize <= 0)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	// Assign value
	cfg->batch = batchSize;

RET:
	return ret;
}

void cnn_config_get_batch_size(cnn_config_t cfg, int* batchPtr)
{
	if(batchPtr != NULL)
	{
		*batchPtr = cfg->batch;
	}
}

int cnn_config_set_layers(cnn_config_t cfg, int layers)
{
	int ret = CNN_NO_ERROR;
	int i, preLayers;

	union CNN_CONFIG_LAYER* tmpLayerCfg;

	// Checking
	if(layers <= 0)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	preLayers = cfg->layers;
	if(preLayers == layers)
	{
		ret = CNN_NO_ERROR;
		goto RET;
	}

	// Reallocate layer config list
	tmpLayerCfg = realloc(cfg->layerCfg, layers * sizeof(union CNN_CONFIG_LAYER));
	if(tmpLayerCfg == NULL)
	{
		ret = CNN_MEM_FAILED;
		goto RET;
	}
	else
	{
		cfg->layerCfg = tmpLayerCfg;
		cfg->layers = layers;
	}

	// Set default layer config
	if(preLayers <= 0)
	{
		i = 1;
		cfg->layerCfg[0].type = CNN_LAYER_INPUT;
	}
	else
	{
		i = preLayers;
	}

	for( ; i < layers; i++)
	{
		ret = cnn_config_set_full_connect(cfg, i, CNN_DEFAULT_FC_SIZE);
		assert(ret == CNN_NO_ERROR);
	}

RET:
	return ret;
}

void cnn_config_get_layers(cnn_config_t cfg, int* layersPtr)
{
	if(layersPtr != NULL)
	{
		*layersPtr = cfg->layers;
	}
}

int cnn_config_get_layer_type(cnn_config_t cfg, int layerIndex, cnn_layer_t* typePtr)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(layerIndex < 0 || layerIndex >= cfg->layers)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	if(typePtr != NULL)
	{
		*typePtr = cfg->layerCfg[layerIndex].type;
	}

RET:
	return ret;
}

int cnn_config_set_full_connect(cnn_config_t cfg, int layerIndex, int size)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(size <= 0 || layerIndex <= 0 || layerIndex >= cfg->layers)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	// Set config
	cfg->layerCfg[layerIndex].type = CNN_LAYER_FC;
	cfg->layerCfg[layerIndex].fc.size = size;

RET:
	return ret;
}

int cnn_config_get_full_connect(cnn_config_t cfg, int layerIndex, int* sizePtr)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(layerIndex < 0 || layerIndex >= cfg->layers)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	if(cfg->layerCfg[layerIndex].type != CNN_LAYER_FC)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	// Assign value
	if(sizePtr != NULL)
	{
		*sizePtr = cfg->layerCfg[layerIndex].fc.size;
	}

RET:
	return ret;
}

int cnn_config_set_activation(cnn_config_t cfg, int layerIndex, cnn_afunc_t aFuncID)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(layerIndex <= 0 || layerIndex >= cfg->layers || aFuncID < 0 || aFuncID >= CNN_AFUNC_AMOUNT)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	// Set config
	cfg->layerCfg[layerIndex].type = CNN_LAYER_AFUNC;
	cfg->layerCfg[layerIndex].aFunc.id = aFuncID;

RET:
	return ret;
}

int cnn_config_get_activation(cnn_config_t cfg, int layerIndex, cnn_afunc_t* idPtr)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(layerIndex < 0 || layerIndex >= cfg->layers)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	if(cfg->layerCfg[layerIndex].type != CNN_LAYER_AFUNC)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	// Assing value
	if(idPtr != NULL)
	{
		*idPtr = cfg->layerCfg[layerIndex].aFunc.id;
	}

RET:
	return ret;
}

int cnn_config_set_convolution(cnn_config_t cfg, int layerIndex, cnn_dim_t convDim, int size)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(layerIndex <= 0 || layerIndex >= cfg->layers ||
			convDim <= 0 || convDim > 2 ||
			size <= 0)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	// Set config
	cfg->layerCfg[layerIndex].type = CNN_LAYER_CONV;
	cfg->layerCfg[layerIndex].conv.dim = convDim;
	cfg->layerCfg[layerIndex].conv.size = size;

RET:
	return ret;
}

int cnn_config_get_convolution(cnn_config_t cfg, int layerIndex, cnn_dim_t* dimPtr,
		int* sizePtr)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(layerIndex < 0 || layerIndex >= cfg->layers)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	if(cfg->layerCfg[layerIndex].type != CNN_LAYER_CONV)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	// Assign value
	if(dimPtr != NULL)
	{
		*dimPtr = cfg->layerCfg[layerIndex].conv.dim;
	}

	if(sizePtr != NULL)
	{
		*sizePtr = cfg->layerCfg[layerIndex].conv.size;
	}

RET:
	return ret;
}

int cnn_config_set_pooling(cnn_config_t cfg, int layerIndex, cnn_dim_t dim, cnn_pool_t type,
		int size)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(layerIndex <= 0 || layerIndex >= cfg->layers ||
			dim <= 0 || dim > 2 || size <= 0)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	// Set config
	cfg->layerCfg[layerIndex].type = CNN_LAYER_POOL;
	cfg->layerCfg[layerIndex].pool.poolType = type;
	cfg->layerCfg[layerIndex].pool.dim = dim;
	cfg->layerCfg[layerIndex].pool.size = size;

RET:
	return ret;
}

int cnn_config_get_pooling(cnn_config_t cfg, int layerIndex, cnn_dim_t* dimPtr,
		cnn_pool_t* typePtr, int* sizePtr)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(layerIndex < 0 || layerIndex >= cfg->layers)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	if(cfg->layerCfg[layerIndex].type != CNN_LAYER_POOL)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	// Assign value
	if(dimPtr != NULL)
	{
		*dimPtr = cfg->layerCfg[layerIndex].pool.dim;
	}

	if(typePtr != NULL)
	{
		*typePtr = cfg->layerCfg[layerIndex].pool.poolType;
	}

	if(sizePtr != NULL)
	{
		*sizePtr = cfg->layerCfg[layerIndex].pool.size;
	}

RET:
	return ret;
}
