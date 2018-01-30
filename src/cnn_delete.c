#include <stdlib.h>
#include <string.h>

#include "cnn.h"
#include "cnn_private.h"

void cnn_delete(cnn_t cnn)
{
	if(cnn != NULL)
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
	cnn_free(matPtr->mat);
	cnn_free(matPtr->grad);

	// Zero memory
	memset(matPtr, 0, sizeof(struct CNN_MAT));
}

void cnn_layer_input_delete(union CNN_LAYER* layerPtr)
{
	// Free memory
	cnn_mat_delete(&layerPtr->outMat.data);

	// Zero memory
	memset(layerPtr, 0, sizeof(union CNN_LAYER));
}

void cnn_layer_afunc_delete(struct CNN_LAYER_AFUNC* layerPtr)
{
	// Free memory
	cnn_mat_delete(&layerPtr->outMat.data);
	cnn_mat_delete(&layerPtr->gradMat);

	// Zero memory
	memset(layerPtr, 0, sizeof(struct CNN_LAYER_AFUNC));
}

void cnn_layer_fc_delete(struct CNN_LAYER_FC* layerPtr)
{
	// Free memory
	cnn_mat_delete(&layerPtr->outMat.data);
	cnn_mat_delete(&layerPtr->weight);
	cnn_mat_delete(&layerPtr->bias);

	// Zero memory
	memset(layerPtr, 0, sizeof(struct CNN_LAYER_FC));
}

void cnn_layer_conv_delete(struct CNN_LAYER_CONV* layerPtr)
{
	// Free memory
	cnn_mat_delete(&layerPtr->outMat.data);
	cnn_mat_delete(&layerPtr->kernel);
	cnn_mat_delete(&layerPtr->bias);

	// Zero memory
	memset(layerPtr, 0, sizeof(struct CNN_LAYER_CONV));
}

void cnn_network_delete(struct CNN* cnn)
{
	int i;

	if(cnn->layerList != NULL)
	{
		// Delete CNN layers
		for(i = 0; i < cnn->cfg.layers; i++)
		{
			switch(cnn->cfg.layerCfg[i].type)
			{
				case CNN_LAYER_INPUT:
					cnn_layer_input_delete(&cnn->layerList[i]);
					break;

				case CNN_LAYER_FC:
					cnn_layer_fc_delete(&cnn->layerList[i].fc);
					break;

				case CNN_LAYER_AFUNC:
					cnn_layer_afunc_delete(&cnn->layerList[i].aFunc);
					break;

				case CNN_LAYER_CONV:
					cnn_layer_conv_delete(&cnn->layerList[i].conv);
					break;
			}
		}

		// Free memory
		cnn_free(cnn->layerList);
	}

	cnn->layerList = NULL;
}

