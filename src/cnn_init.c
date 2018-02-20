#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "cnn.h"
#include "cnn_private.h"
#include "cnn_init.h"

float cnn_normal_distribution(struct CNN_BOX_MULLER* bmPtr, double mean, double stddev)
{
	double dPI = 2 * M_PI;
	double u0, u1;
	double z0, z1;
	double calcTmp;

	// Inverse saved
	bmPtr->saved = !(bmPtr->saved);

	if(!bmPtr->saved)
	{
		return bmPtr->val * stddev + mean;
	}

	// Generate
	do
	{
		u0 = (double)rand() / (double)RAND_MAX;
		u1 = (double)rand() / (double)RAND_MAX;
		calcTmp = u0 * u0 + u1 * u1;
	} while(calcTmp > 1.0 || calcTmp <= 0.0);

	z0 = sqrt(-2.0 * log(u0)) * cos(dPI * u1);
	z1 = sqrt(-2.0 * log(u0)) * sin(dPI * u1);
	bmPtr->val = z1;

	return z0 * stddev + mean;
}

float cnn_xavier_init(struct CNN_BOX_MULLER* bmPtr, int inSize, int outSize)
{
	double var;

	// Xavier initialization
	var = 2.0 / (double)(inSize + outSize);

	return cnn_normal_distribution(bmPtr, 0.0, sqrt(var));
}

void cnn_rand_network(cnn_t cnn)
{
	int i, j;
	size_t size;
	struct CNN_CONFIG* cfgRef;

	struct CNN_BOX_MULLER bm;

	srand(time(NULL));

	// Get reference
	cfgRef = &cnn->cfg;

	// Rand network
	for(i = 1; i < cfgRef->layers; i++)
	{
		// Setup random method
		memset(&bm, 0, sizeof(struct CNN_BOX_MULLER));

		switch(cfgRef->layerCfg[i].type)
		{
			case CNN_LAYER_FC:
				// Random weight
				size = cnn->layerList[i].fc.weight.rows * cnn->layerList[i].fc.weight.cols;
				for(j = 0; j < size; j++)
				{
					cnn->layerList[i].fc.weight.mat[j] =
						cnn_xavier_init(&bm, cnn->layerList[i - 1].outMat.data.cols,
								cnn->layerList[i].outMat.data.cols);
				}

				// Zero bias
				size = cnn->layerList[i].fc.bias.rows * cnn->layerList[i].fc.weight.cols;
				memset(cnn->layerList[i].fc.bias.mat, 0, size * sizeof(float));

				break;

			case CNN_LAYER_CONV:
				// Random kernel
				size = cnn->layerList[i].conv.kernel.rows * cnn->layerList[i].conv.kernel.cols;
				for(j = 0; j < size; j++)
				{
					cnn->layerList[i].conv.kernel.mat[j] =
						cnn_xavier_init(&bm, cnn->layerList[i - 1].outMat.data.cols,
								cnn->layerList[i].outMat.data.cols);
				}

				// Zero bias
				size = cnn->layerList[i].conv.bias.rows * cnn->layerList[i].conv.bias.cols;
				memset(cnn->layerList[i].conv.bias.mat, 0, size * sizeof(float));

				break;
		}
	}
}

void cnn_zero_network(cnn_t cnn)
{
	int i;
	size_t size;
	struct CNN_CONFIG* cfgRef;

	// Get reference
	cfgRef = &cnn->cfg;

	// Rand network
	for(i = 1; i < cfgRef->layers; i++)
	{
		switch(cfgRef->layerCfg[i].type)
		{
			case CNN_LAYER_FC:
				// Zero weight
				size = cnn->layerList[i].fc.weight.rows * cnn->layerList[i].fc.weight.cols;
				memset(cnn->layerList[i].fc.weight.mat, 0, size * sizeof(float));

				// Zero bias
				size = cnn->layerList[i].fc.bias.rows * cnn->layerList[i].fc.weight.cols;
				memset(cnn->layerList[i].fc.bias.mat, 0, size * sizeof(float));

				break;

			case CNN_LAYER_CONV:
				// Zero kernel
				size = cnn->layerList[i].conv.kernel.rows * cnn->layerList[i].conv.kernel.cols;
				memset(cnn->layerList[i].conv.kernel.mat, 0, size * sizeof(float));

				// Zero bias
				size = cnn->layerList[i].conv.bias.rows * cnn->layerList[i].conv.bias.cols;
				memset(cnn->layerList[i].conv.bias.mat, 0, size * sizeof(float));

				break;
		}
	}
}
