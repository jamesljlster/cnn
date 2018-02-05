#include <cblas.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn.h"
#include "cnn_private.h"
#include "cnn_builtin_math.h"
#include "cnn_calc.h"

void cnn_bp(cnn_t cnn, float lRate, float* errGrad)
{
	int i, j, k;
	int srcShift, dstShift;

	float* srcPtr;
	float* dstPtr;

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
				// Find weight delta matrix
				cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
						layerRef[i].fc.weight.rows,
						layerRef[i].fc.weight.cols,
						cfgRef->batch,
						1.0,
						layerRef[i - 1].outMat.data.mat, layerRef[i - 1].outMat.data.cols,
						layerRef[i].outMat.data.grad, layerRef[i].outMat.data.cols, 0.0,
						layerRef[i].fc.weight.grad, layerRef[i].fc.weight.cols);

				// Find bias delta matrix
				memset(cnn->layerList[i].fc.bias.grad, 0, sizeof(float) *
						cnn->layerList[i].fc.bias.cols);

				for(j = 0; j < cfgRef->batch; j++)
				{
					srcShift = j * cnn->layerList[i].outMat.data.cols;
					cblas_saxpy(cnn->layerList[i].fc.bias.cols, 1.0,
							&cnn->layerList[i].outMat.data.grad[srcShift],
							1, cnn->layerList[i].fc.bias.grad, 1);
				}

				// Find layer gradient
				if(i > 1)
				{
					cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
							layerRef[i - 1].outMat.data.rows,
							layerRef[i - 1].outMat.data.cols,
							layerRef[i].outMat.data.cols,
							1.0,
							layerRef[i].outMat.data.grad, layerRef[i].outMat.data.cols,
							layerRef[i].fc.weight.mat, layerRef[i].fc.weight.cols, 0.0,
							layerRef[i - 1].outMat.data.grad, layerRef[i - 1].outMat.data.cols);
				}

				// Update weight
				cblas_saxpy(cnn->layerList[i].fc.weight.rows * cnn->layerList[i].fc.weight.cols,
						lRate,
						cnn->layerList[i].fc.weight.grad, 1,
						cnn->layerList[i].fc.weight.mat, 1);

				// Update bias
				cblas_saxpy(cnn->layerList[i].fc.bias.cols, lRate,
						cnn->layerList[i].fc.bias.grad, 1,
						cnn->layerList[i].fc.bias.mat, 1);

				break;

			// Activation function
			case CNN_LAYER_AFUNC:
				if(i > 1)
				{
					for(j = 0; j < cfgRef->batch; j++)
					{
						srcShift = j * layerRef[i - 1].outMat.data.cols;
						if(cfgRef->layerCfg[i].aFunc.id == CNN_SOFTMAX)
						{
							dstShift = srcShift * layerRef[i].outMat.data.cols;
						}
						else
						{
							dstShift = srcShift;
						}

						srcPtr = &layerRef[i - 1].outMat.data.mat[srcShift];
						dstPtr = &layerRef[i].aFunc.gradMat.mat[dstShift];

						// Find gradient matrix
						cnn_afunc_grad_list[cfgRef->layerCfg[i].aFunc.id](dstPtr,
								srcPtr, layerRef[i].outMat.data.cols,
								&layerRef[i].aFunc.buf.mat[srcShift]);

						// Find layer gradient
						if(cfgRef->layerCfg[i].aFunc.id == CNN_SOFTMAX)
						{
							cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
									1, layerRef[i - 1].outMat.data.cols,
									layerRef[i].outMat.data.cols,
									1.0,
									&layerRef[i].outMat.data.grad[srcShift],
									layerRef[i].outMat.data.cols,
									dstPtr, layerRef[i].aFunc.gradMat.cols, 0.0,
									&layerRef[i - 1].outMat.data.grad[srcShift],
									layerRef[i - 1].outMat.data.cols);
						}
						else
						{
							for(k = 0; k < layerRef[i - 1].outMat.data.cols; k++)
							{
								layerRef[i - 1].outMat.data.grad[srcShift + k] = dstPtr[k] *
									layerRef[i].outMat.data.grad[srcShift + k];
							}
						}
					}
				}

				break;

			// Convolution
			case CNN_LAYER_CONV:
				// Find kernel delta matrix
				memset(cnn->layerList[i].conv.kernel.grad, 0, sizeof(float) *
						cnn->layerList[i].conv.kernel.rows *
						cnn->layerList[i].conv.kernel.cols);

				for(j = 0; j < cfgRef->batch; j++)
				{
					srcShift = j * cnn->layerList[i].outMat.data.cols;
					dstShift = j * cnn->layerList[i - 1].outMat.data.cols;

					cnn_conv_2d_kernel_grad((&cnn->layerList[i].outMat.data.grad[srcShift]),
							cnn->layerList[i].outMat.height,
							cnn->layerList[i].outMat.width,
							cnn->layerList[i].conv.kernel.grad,
							cnn->layerList[i].conv.kernel.cols,
							cnn->layerList[i].conv.inChannel,
							(&cnn->layerList[i - 1].outMat.data.mat[dstShift]),
							cnn->layerList[i - 1].outMat.height,
							cnn->layerList[i - 1].outMat.width);
				}

				// Find bias delta matrix
				memset(cnn->layerList[i].conv.bias.grad, 0, sizeof(float) *
						cnn->layerList[i].conv.bias.cols);

				for(j = 0; j < cfgRef->batch; j++)
				{
					srcShift = j * cnn->layerList[i].outMat.data.cols;
					cblas_saxpy(cnn->layerList[i].conv.bias.cols, 1.0,
							&cnn->layerList[i].outMat.data.grad[srcShift],
							1, cnn->layerList[i].conv.bias.grad, 1);
				}

				// Find layer gradient
				if(i > 1)
				{
					memset(cnn->layerList[i - 1].outMat.data.grad, 0, sizeof(float) *
							cnn->layerList[i - 1].outMat.data.rows *
							cnn->layerList[i - 1].outMat.data.cols);

					for(j = 0; j < cfgRef->batch; j++)
					{
						srcShift = j * cnn->layerList[i].outMat.data.cols;
						dstShift = j * cnn->layerList[i - 1].outMat.data.cols;

						cnn_conv_2d_grad((&cnn->layerList[i - 1].outMat.data.grad[dstShift]),
								cnn->layerList[i - 1].outMat.height,
								cnn->layerList[i - 1].outMat.width,
								cnn->layerList[i].conv.kernel.mat,
								cnn->layerList[i].conv.kernel.cols,
								cnn->layerList[i].conv.inChannel,
								(&cnn->layerList[i].outMat.data.grad[srcShift]),
								cnn->layerList[i].outMat.height,
								cnn->layerList[i].outMat.width);
					}
				}

				// Update kernel
				cblas_saxpy(cnn->layerList[i].conv.kernel.cols *
							cnn->layerList[i].conv.kernel.rows, lRate,
						cnn->layerList[i].conv.kernel.grad, 1,
						cnn->layerList[i].conv.kernel.mat, 1);

				// Update bias
				cblas_saxpy(cnn->layerList[i].conv.bias.cols, lRate,
						cnn->layerList[i].conv.bias.grad, 1,
						cnn->layerList[i].conv.bias.mat, 1);

				break;

			// Pooling
			case CNN_LAYER_POOL:
				if(i > 1)
				{
					// Zero layer gradient
					memset(layerRef[i - 1].outMat.data.grad, 0, sizeof(float) *
							layerRef[i - 1].outMat.data.rows *
							layerRef[i - 1].outMat.data.cols);

					for(j = 0; j < cfgRef->batch; j++)
					{
						srcShift = j * layerRef[i].outMat.data.cols;
						dstShift = j * layerRef[i - 1].outMat.data.cols;

						srcPtr = &layerRef[i].outMat.data.grad[srcShift];

						// Find layer gradient
							cnn_pool_2d_max_grad((&layerRef[i - 1].outMat.data.grad[dstShift]),
									(&layerRef[i].pool.indexMat[srcShift]),
									srcPtr, layerRef[i].outMat.height, layerRef[i].outMat.width);
					}
				}

				break;

			default:
				assert((cfgRef->layerCfg[i].type) > 0 &&
						(cfgRef->layerCfg[i].type <= CNN_LAYER_POOL));
		}
	}
}

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
							cnn->layerList[i].conv.inChannel,
							srcPtr, layerRef[i - 1].outMat.height, layerRef[i - 1].outMat.width);
				}

				break;

			// Pooling
			case CNN_LAYER_POOL:
				for(j = 0; j < cfgRef->batch; j++)
				{
					srcShift = j * layerRef[i - 1].outMat.data.cols;
					dstShift = j * layerRef[i].outMat.data.cols;

					srcPtr = &layerRef[i - 1].outMat.data.mat[srcShift];
					dstPtr = &layerRef[i].outMat.data.mat[dstShift];

					cnn_pool_2d_max(dstPtr, (&layerRef[i].pool.indexMat[dstShift]),
							layerRef[i].outMat.height, layerRef[i].outMat.width,
							cfgRef->layerCfg[i].pool.size, srcPtr,
							layerRef[i - 1].outMat.height, layerRef[i - 1].outMat.width);
				}

				break;

			default:
				assert((cfgRef->layerCfg[i].type) > 0 &&
						(cfgRef->layerCfg[i].type <= CNN_LAYER_POOL));
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
