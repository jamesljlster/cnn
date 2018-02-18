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

float cnn_float_rand()
{
	if(__cnnRandInit <= 0)
	{
		//srand((unsigned int)time(NULL));
		__cnnRandInit = 1;
	}

	return (float)(rand() % 4096) / 4096.0f;
}

float cnn_rand(int inSize)
{
	float stddev;

	// Xavier initialization
	stddev = 2.0 / (float)inSize;

	return (cnn_float_rand() - 0.499821f) / 0.287194f * stddev;
}

float cnn_zero(int inSize)
{
	return 0;
}

void cnn_init_network(cnn_t cnn, float (*initMethod)(int inSize))
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
					cnn->layerList[i].fc.weight.mat[j] =
						initMethod(cnn->layerList[i - 1].outMat.data.cols);
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
					cnn->layerList[i].conv.kernel.mat[j] = initMethod(size);
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

