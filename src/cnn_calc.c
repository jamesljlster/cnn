#include <cblas.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn.h"
#include "cnn_private.h"
#include "cnn_builtin_math.h"
#include "cnn_calc.h"

void cnn_forward(cnn_t cnn, float* inputMat, float* outputMat)
{
	int i, j;
	int srcShift, dstShift;

	float* srcPtr;
	float* dstPtr;

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
				// Weight matrix multiplication
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						layerRef[i - 1].outMat.data.rows,
						layerRef[i].outMat.data.cols,
						layerRef[i - 1].outMat.data.cols,
						1.0,
						layerRef[i - 1].outMat.data.mat, layerRef[i - 1].outMat.data.cols,
						layerRef[i].fc.weight.mat, layerRef[i].fc.weight.cols,
						0.0, layerRef[i].outMat.data.mat, layerRef[i].outMat.data.cols);

				// Add bias
				for(j = 0; j < cfgRef->batch; j++)
				{
					cblas_saxpy(layerRef[i].fc.bias.cols, 1.0,
							layerRef[i].fc.bias.mat, 1,
							&layerRef[i].outMat.data.mat[j * layerRef[i].outMat.data.cols], 1);
				}

				break;

			// Activation function
			case CNN_LAYER_AFUNC:
				for(j = 0; j < cfgRef->batch; j++)
				{
					srcShift = j * layerRef[i - 1].outMat.data.cols;
					dstShift = j * layerRef[i].outMat.data.cols;

					srcPtr = &layerRef[i - 1].outMat.data.mat[srcShift];
					dstPtr = &layerRef[i].outMat.data.mat[dstShift];

					cnn_afunc_list[cfgRef->layerCfg[i].aFunc.id](dstPtr,
							srcPtr, layerRef[i].outMat.data.cols, NULL);
				}

				break;

			// Convolution
			case CNN_LAYER_CONV:
				for(j = 0; j < cfgRef->batch; j++)
				{
					srcShift = j * layerRef[i - 1].outMat.data.cols;
					dstShift = j * layerRef[i].outMat.data.cols;

					srcPtr = &layerRef[i - 1].outMat.data.mat[srcShift];
					dstPtr = &layerRef[i].outMat.data.mat[dstShift];

					cnn_conv_2d(dstPtr, layerRef[i].outMat.height, layerRef[i].outMat.width,
							layerRef[i].conv.kernel.mat, cfgRef->layerCfg[i].conv.size,
							srcPtr, layerRef[i - 1].outMat.height, layerRef[i - 1].outMat.width);
				}

				break;

			default:
				assert((cfgRef->layerCfg[i].type) > 0 &&
						(cfgRef->layerCfg[i].type <= CNN_LAYER_CONV));
		}
	}

	// Copy output
	if(outputMat != NULL)
	{
		memcpy(outputMat, layerRef[i].outMat.data.mat, sizeof(float) *
				layerRef[i].outMat.data.rows * layerRef[i].outMat.data.cols);
	}
}
