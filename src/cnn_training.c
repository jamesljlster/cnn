#include <stdlib.h>
#include <string.h>

#include "cnn.h"
#include "cnn_private.h"

int cnn_training(cnn_t cnn, float* inputMat, float* desireMat, float* outputMat, float* errMat)
{
	return cnn_training_custom(cnn, cnn->cfg.lRate, inputMat, desireMat, outputMat, errMat);
}

int cnn_training_custom(cnn_t cnn, float lRate, float* inputMat, float* desireMat, float* outputMat, float* errMat)
{
	int i;
	int outSize;
	int ret = CNN_NO_ERROR;

	float* outStore = NULL;
	float* errStore = NULL;

	struct CNN_CONFIG* cfgRef;

	// Get reference
	cfgRef = &cnn->cfg;
	outSize = cnn->layerList[cfgRef->layers - 1].outMat.data.rows *
		cnn->layerList[cfgRef->layers - 1].outMat.data.cols;

	// Memory allocation
	cnn_alloc(outStore, outSize, float, ret, RET);
	cnn_alloc(errStore, outSize, float, ret, RET);

	// Forward
	cnn_forward(cnn, inputMat, outStore);

	// Find error
	for(i = 0; i < outSize; i++)
	{
		errStore[i] = desireMat[i] - outStore[i];
	}

	// Backpropagation
	cnn_bp(cnn, lRate, errStore);

	if(outputMat != NULL)
	{
		memcpy(outputMat, outStore, outSize * sizeof(float));
	}

	if(errMat != NULL)
	{
		memcpy(errMat, errStore, outSize * sizeof(float));
	}

RET:
	cnn_free(outStore);
	cnn_free(errStore);

	return ret;
}

