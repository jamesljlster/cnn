#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn.h"
#include "cnn_private.h"
#include "cnn_builtin_math.h"

#define CNN_DEFAULT_INPUT_WIDTH 32
#define CNN_DEFAULT_INPUT_HEIGHT 32
#define CNN_DEFAULT_OUTPUTS 2
#define CNN_DEFAULT_LAYERS 2
#define CNN_DEFAULT_AFUNC CNN_SOFTMAX
#define CNN_DEFAULT_FC_SIZE 16

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

	// Set default settings
	cnn_run(cnn_config_set_input_size(cfg, CNN_DEFAULT_INPUT_WIDTH, CNN_DEFAULT_INPUT_HEIGHT), ret, RET);
	cnn_run(cnn_config_set_outputs(cfg, CNN_DEFAULT_OUTPUTS), ret, RET);
	cnn_run(cnn_config_set_layers(cfg, CNN_DEFAULT_LAYERS), ret, RET);
	cnn_run(cnn_config_set_full_connect(cfg, 0, CNN_DEFAULT_FC_SIZE), ret, RET);
	cnn_run(cnn_config_set_activation(cfg, 1, CNN_DEFAULT_AFUNC), ret, RET);

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

int cnn_config_set_input_size(cnn_config_t cfg, int width, int height)
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

RET:
	return ret;
}

int cnn_config_set_outputs(cnn_config_t cfg, int outputs)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(outputs <= 0)
	{
		ret = CNN_INVALID_ARG;
		goto RET;
	}

	// Assing value
	cfg->outputs = outputs;

RET:
	return ret;
}

int cnn_set_layers(cnn_config_t cfg, int layers)
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
	i = (preLayers > 0) ? preLayers - 1 : 0;
	for( ; i < layers; i++)
	{
		ret = cnn_config_set_full_connect(cfg, i, CNN_DEFAULT_FC_SIZE);
		assert(ret == CNN_NO_ERROR);
	}

RET:
	return ret;
}

int cnn_config_set_full_connect(cnn_config_t cfg, int layerIndex, int size)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(size <= 0 || layerIndex < 0 || layerIndex >= cfg->layers)
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

int cnn_config_set_activation(cnn_config_t cfg, int layerIndex, int aFuncID)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(layerIndex < 0 || layerIndex >= cfg->layers || aFuncID < 0 || aFuncID >= CNN_AFUNC_AMOUNT)
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

int cnn_config_set_convolution(cnn_config_t cfg, int layerIndex, int convDim, int size)
{
	int ret = CNN_NO_ERROR;

	// Checking
	if(layerIndex < 0 || layerIndex >= cfg->layers ||
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

