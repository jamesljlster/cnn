#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "cnn.h"
#include "cnn_private.h"

#define NUM_PRECISION 1000
#define NUM_MAX 1
#define NUM_MIN -1

int __cnnRandInit = 0;

float cnn_rand(void)
{
	int randRange;

	if(__cnnRandInit <= 0)
	{
		srand((unsigned int)time(NULL));
		__cnnRandInit = 1;
	}

	randRange = (NUM_MAX - NUM_MIN) * NUM_PRECISION + 1;

	return (float)(rand() % randRange) / (float)(NUM_PRECISION) + (float)NUM_MIN;
}

float cnn_zero(void)
{
	return 0;
}

void cnn_init_network(cnn_t cnn, float (*initMethod)(void))
{
	int i, j;
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
				// Random weight
				size = cnn->layerList[i].fc.weight.rows * cnn->layerList[i].fc.weight.cols;
				for(j = 0; j < size; j++)
				{
					cnn->layerList[i].fc.weight.mat[j] = initMethod();
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
					cnn->layerList[i].conv.kernel.mat[j] = initMethod();
				}

				// Zero bias
				size = cnn->layerList[i].conv.bias.rows * cnn->layerList[i].conv.bias.cols;
				memset(cnn->layerList[i].conv.bias.mat, 0, size * sizeof(float));

				break;
		}
	}
}

void cnn_rand_network(cnn_t cnn)
{
	cnn_init_network(cnn, cnn_rand);
}

void cnn_zero_network(cnn_t cnn)
{
	cnn_init_network(cnn, cnn_zero);
}

