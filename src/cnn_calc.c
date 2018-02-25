#include <cblas.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn_private.h"
#include "cnn_calc.h"

void cnn_update(cnn_t cnn, float lRate)
{
	int i;
	struct CNN_CONFIG* cfgRef;
	union CNN_LAYER* layerRef;

	// Set reference
	layerRef = cnn->layerList;
	cfgRef = &cnn->cfg;

	// Update network and clear gradient
	for(i = cfgRef->layers - 1; i > 0; i--)
	{
		// Clear layer gradient
		memset(layerRef[i].outMat.data.grad, 0, sizeof(float) *
				layerRef[i].outMat.data.rows * layerRef[i].outMat.data.cols);

		switch(cfgRef->layerCfg[i].type)
		{
			// Fully connected
			case CNN_LAYER_FC:
				// Update weight
				cblas_saxpy(layerRef[i].fc.weight.rows * layerRef[i].fc.weight.cols,
						lRate,
						layerRef[i].fc.weight.grad, 1,
						layerRef[i].fc.weight.mat, 1);

				// Update bias
				cblas_saxpy(layerRef[i].fc.bias.cols, lRate,
						layerRef[i].fc.bias.grad, 1,
						layerRef[i].fc.bias.mat, 1);

				// Clear gradient
				memset(layerRef[i].fc.weight.grad, 0, sizeof(float) *
						layerRef[i].fc.weight.rows * layerRef[i].fc.weight.cols);
				memset(layerRef[i].fc.bias.grad, 0, sizeof(float) *
						layerRef[i].fc.bias.rows * layerRef[i].fc.bias.cols);

				break;

			// Convolution
			case CNN_LAYER_CONV:
				// Update kernel
				cblas_saxpy(layerRef[i].conv.kernel.cols * layerRef[i].conv.kernel.rows,
						lRate, layerRef[i].conv.kernel.grad, 1,
						layerRef[i].conv.kernel.mat, 1);

				// Update bias
				cblas_saxpy(layerRef[i].conv.bias.cols, lRate,
						layerRef[i].conv.bias.grad, 1,
						layerRef[i].conv.bias.mat, 1);

				// Clear gradient
				memset(layerRef[i].conv.kernel.grad, 0, sizeof(float) *
						layerRef[i].conv.kernel.rows * layerRef[i].conv.kernel.cols);
				memset(layerRef[i].conv.bias.grad, 0, sizeof(float) *
						layerRef[i].conv.bias.rows * layerRef[i].conv.bias.cols);

				break;
		}
	}
}

void cnn_backward(cnn_t cnn, float* errGrad)
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
				cnn_backward_fc(layerRef, cfgRef, i);
				break;

			// Activation function
			case CNN_LAYER_AFUNC:
				cnn_backward_afunc(layerRef, cfgRef, i);
				break;

			// Convolution
			case CNN_LAYER_CONV:
				cnn_backward_conv(layerRef, cfgRef, i);
				break;

			// Pooling
			case CNN_LAYER_POOL:
				cnn_backward_pool(layerRef, cfgRef, i);
				break;

			// Dropout
			case CNN_LAYER_DROP:
				cnn_backward_drop(layerRef, cfgRef, i);
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
