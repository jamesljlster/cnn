#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn.h"
#include "cnn_private.h"

void cnn_get_input_size(cnn_t cnn, int* wPtr, int* hPtr, int* cPtr)
{
	// Assign value
	if(wPtr != NULL)
	{
		*wPtr = cnn->cfg.width;
	}

	if(hPtr != NULL)
	{
		*hPtr = cnn->cfg.height;
	}

	if(cPtr != NULL)
	{
		*cPtr = cnn->cfg.channel;
	}
}

void cnn_get_output_size(cnn_t cnn, int* wPtr, int* hPtr, int* cPtr)
{
	int index = cnn->cfg.layers - 1;

	// Assign value
	if(wPtr != NULL)
	{
		*wPtr = cnn->layerList[index].outMat.width;
	}

	if(hPtr != NULL)
	{
		*hPtr = cnn->layerList[index].outMat.height;
	}

	if(cPtr != NULL)
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

	// Clone weight and bias
	for(i = 1; i < cfgRef->layers; i++)
	{
		switch(cfgRef->layerCfg[i].type)
		{
			case CNN_LAYER_FC:
				// Clone weight
				memcpy(dstLayerList[i].fc.weight.mat, srcLayerList[i].fc.weight.mat,
						sizeof(float) * srcLayerList[i].fc.weight.rows *
							srcLayerList[i].fc.weight.cols);

				// Clone bias
				memcpy(dstLayerList[i].fc.bias.mat, srcLayerList[i].fc.bias.mat,
						sizeof(float) * srcLayerList[i].fc.bias.rows *
							srcLayerList[i].fc.bias.cols);
				break;

			case CNN_LAYER_CONV:
				// Clone kernel
				memcpy(dstLayerList[i].conv.kernel.mat, srcLayerList[i].conv.kernel.mat,
						sizeof(float) * srcLayerList[i].conv.kernel.rows *
							srcLayerList[i].conv.kernel.cols);

				// Clone bias
#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
				memcpy(dstLayerList[i].conv.bias.mat, srcLayerList[i].conv.bias.mat,
						sizeof(float) * srcLayerList[i].conv.bias.rows *
							srcLayerList[i].conv.bias.cols);
#endif
				break;
		}
	}
}
