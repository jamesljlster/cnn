#include <cblas.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn_private.h"
#include "cnn_calc.h"

void cnn_bp(cnn_t cnn, float lRate, float* errGrad)
{
	int i;

	struct CNN_CONFIG* cfgRef;
	union CNN_LAYER* layerRef;

	// Set reference
	layerRef = cnn->layerList;
	cfgRef = &cnn->cfg;

	// Copy gradient vector
	memcpy(layerRef[cfgRef->layers - 1].outMat.data.grad, errGrad, sizeof(float) *
			layerRef[cfgRef->layers - 1].outMat.data.rows *
			layerRef[cfgRef->layers - 1].outMat.data.cols);

	// Backpropagation
	for(i = cfgRef->layers - 1; i > 0; i--)
	{
		switch(cfgRef->layerCfg[i].type)
		{
			// Fully connected
			case CNN_LAYER_FC:
				cnn_bp_fc(layerRef, cfgRef, i, lRate);
				break;

			// Activation function
			case CNN_LAYER_AFUNC:
				cnn_bp_afunc(layerRef, cfgRef, i, lRate);
				break;

			// Convolution
			case CNN_LAYER_CONV:
				cnn_bp_conv(layerRef, cfgRef, i, lRate);
				break;

			// Pooling
			case CNN_LAYER_POOL:
				cnn_bp_pool(layerRef, cfgRef, i, lRate);
				break;

			// Dropout
			case CNN_LAYER_DROP:
				cnn_bp_drop(layerRef, cfgRef, i, lRate);
				break;

			default:
				assert(!"Invalid layer type");
		}
	}
}

void cnn_forward(cnn_t cnn, float* inputMat, float* outputMat)
{
	int i;

	struct CNN_CONFIG* cfgRef;
	union CNN_LAYER* layerRef;

	// Set reference
	layerRef = cnn->layerList;
	cfgRef = &cnn->cfg;

	// Copy input
	memcpy(layerRef[0].outMat.data.mat, inputMat, sizeof(float) *
			layerRef[0].outMat.data.rows * layerRef[0].outMat.data.cols);

	// Forward computation
	for(i = 1; i < cfgRef->layers; i++)
	{
		switch(cfgRef->layerCfg[i].type)
		{
			// Fully connected
			case CNN_LAYER_FC:
				cnn_forward_fc(layerRef, cfgRef, i);
				break;

			// Activation function
			case CNN_LAYER_AFUNC:
				cnn_forward_afunc(layerRef, cfgRef, i);
				break;

			// Convolution
			case CNN_LAYER_CONV:
				cnn_forward_conv(layerRef, cfgRef, i);
				break;

			// Pooling
			case CNN_LAYER_POOL:
				cnn_forward_pool(layerRef, cfgRef, i);
				break;

			// Dropout
			case CNN_LAYER_DROP:
				if(cnn->dropEnable)
				{
					cnn_forward_drop(layerRef, cfgRef, i);
				}
				else
				{
					memcpy(layerRef[i].outMat.data.mat, layerRef[i - 1].outMat.data.mat,
							sizeof(float) * layerRef[i].outMat.data.rows *
							layerRef[i].outMat.data.cols);
				}

				break;

			default:
				assert(!"Invalid layer type");
		}
	}

	// Copy output
	if(outputMat != NULL)
	{
		memcpy(outputMat, layerRef[cfgRef->layers - 1].outMat.data.mat, sizeof(float) *
				layerRef[cfgRef->layers - 1].outMat.data.rows *
				layerRef[cfgRef->layers - 1].outMat.data.cols);
	}
}
