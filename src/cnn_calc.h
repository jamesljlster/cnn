#ifndef __CNN_CALC_H__
#define __CNN_CALC_H__

#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#include "cnn.h"
#include "cnn_types.h"
#include "cnn_builtin_math.h"

inline void cnn_drop(float* dst, float* src, int* mask, int size)
{
	for(int __i = 0; __i < size; __i++)
	{
		if(mask[__i] > 0)
		{
			dst[__i] = src[__i];
		}
	}
}

inline void cnn_drop_grad(float* gradDst, float* gradSrc, int* mask, int size)
{
	for(int __i = 0; __i < size; __i++)
	{
		if(mask[__i] > 0)
		{
			gradDst[__i] = gradSrc[__i];
		}
	}
}

inline void cnn_conv_2d(float* dst, int dstRows, int dstCols,
		float* kernel, int kernelSize, int channel, float* src, int srcRows, int srcCols)
{
	for(int __row = 0; __row < dstRows; __row++)
	{
		for(int __col = 0; __col < dstCols; __col++)
		{
			float __conv = 0;
			for(int __ch = 0; __ch < channel; __ch++)
			{
				for(int __convRow = 0; __convRow < kernelSize; __convRow++)
				{
					for(int __convCol = 0; __convCol < kernelSize; __convCol++)
					{
						__conv += kernel[__convRow * kernelSize + __convCol] *
							src[(__row + __convRow) * srcCols + (__col + __convCol + __ch)];
					}
				}
			}
			dst[__row * dstCols + __col] = __conv;
		}
	}
}

inline void cnn_conv_2d_grad(float* grad, int gradRows, int gradCols,
		float* kernel, int kSize, int channel, float* iGrad, int iGradRows, int iGradCols)
{
	for(int __row = 0; __row < iGradRows; __row++)
	{
		for(int __col = 0; __col < iGradCols; __col++)
		{
			for(int __ch = 0; __ch < channel; __ch++)
			{
				for(int __convRow = 0; __convRow < kSize; __convRow++)
				{
					for(int __convCol = 0; __convCol < kSize; __convCol++)
					{
						grad[(__row + __convRow) * gradCols + (__col + __convCol + __ch)] +=
							kernel[__convRow * kSize + __convCol] *
							iGrad[__row * iGradCols + __col];
					}
				}
			}
		}
	}
}

inline void cnn_conv_2d_kernel_grad(float* grad, int gradRows, int gradCols,
		float* kGrad, int kSize, int channel, float* src, int srcRows, int srcCols)
{
	for(int __row = 0; __row < gradRows; __row++)
	{
		for(int __col = 0; __col < gradCols; __col++)
		{
			for(int __ch = 0; __ch < channel; __ch++)
			{
				for(int __convRow = 0; __convRow < kSize; __convRow++)
				{
					for(int __convCol = 0; __convCol < kSize; __convCol++)
					{
						kGrad[__convRow * kSize + __convCol] += grad[__row * gradCols + __col] *
							src[(__row + __convRow) * srcCols + (__col + __convCol + __ch)];
					}
				}
			}
		}
	}
}

inline void cnn_pool_2d_max(float* dst, int* indexMat, int dstRows, int dstCols, int poolSize,
		float* src, int srcRows, int srcCols)
{
	for(int __row = 0; __row < dstRows; __row++)
	{
		for(int __col = 0; __col < dstCols; __col++)
		{
			float __tmp, __max;
			int __maxIndex, __index;
			int __rowShift = __row * poolSize;
			int __colShift = __col * poolSize;
			__max = src[__rowShift * srcCols + __colShift];
			__maxIndex = __rowShift * srcCols + __colShift;
			for(int __poolRow = 0; __poolRow < poolSize; __poolRow++)
			{
				for(int __poolCol = 0; __poolCol < poolSize; __poolCol++)
				{
					__index = (__rowShift + __poolRow) * srcCols + (__colShift + __poolCol);
					__tmp = src[__index];
					if(__tmp > __max)
					{
						__max = __tmp;
						__maxIndex = __index;
					}
				}
			}
			dst[__row * dstCols + __col] = __max;
			indexMat[__row * dstCols + __col] = __maxIndex;
		}
	}
}

inline void cnn_pool_2d_max_grad(float* grad, int* indexMat,
		float* iGrad, int iGradRows, int iGradCols)
{
	for(int __i = 0; __i < iGradRows * iGradCols; __i++)
	{
		grad[indexMat[__i]] = iGrad[__i];
	}
}

inline void cnn_forward_fc(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef, int layerIndex)
{
	// Weight matrix multiplication
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			layerRef[layerIndex - 1].outMat.data.rows,
			layerRef[layerIndex].outMat.data.cols,
			layerRef[layerIndex - 1].outMat.data.cols,
			1.0,
			layerRef[layerIndex - 1].outMat.data.mat, layerRef[layerIndex - 1].outMat.data.cols,
			layerRef[layerIndex].fc.weight.mat, layerRef[layerIndex].fc.weight.cols,
			0.0, layerRef[layerIndex].outMat.data.mat, layerRef[layerIndex].outMat.data.cols);

	// Add bias
	for(int j = 0; j < cfgRef->batch; j++)
	{
		int dstIndex = j * layerRef[layerIndex].outMat.data.cols;
		cblas_saxpy(layerRef[layerIndex].fc.bias.cols, 1.0,
				layerRef[layerIndex].fc.bias.mat, 1,
				&layerRef[layerIndex].outMat.data.mat[dstIndex], 1);
	}
}

inline void cnn_forward_drop(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef,
		int layerIndex)
{
	int size = layerRef[layerIndex].outMat.data.cols;
	int* mask = layerRef[layerIndex].drop.mask;
	float rate = cfgRef->layerCfg[layerIndex].drop.rate;

	// Zero output memory
	memset(layerRef[layerIndex].outMat.data.mat, 0, sizeof(float) *
			layerRef[layerIndex].outMat.data.rows *
			layerRef[layerIndex].outMat.data.cols);

	// Generate dropout mask
	for(int j = 0; j < size; j++)
	{
		if((float)rand() / (float)RAND_MAX >= rate)
		{
			mask[j] = 1;
		}
		else
		{
			mask[j] = 0;
		}
	}

	for(int j = 0; j < cfgRef->batch; j++)
	{
		int srcShift = j * layerRef[layerIndex - 1].outMat.data.cols;
		int dstShift = j * layerRef[layerIndex].outMat.data.cols;

		float* srcPtr = &layerRef[layerIndex - 1].outMat.data.mat[srcShift];
		float* dstPtr = &layerRef[layerIndex].outMat.data.mat[dstShift];

		cnn_drop(dstPtr, srcPtr, mask, size);
	}
}

inline void cnn_forward_afunc(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef,
		int layerIndex)
{
	for(int j = 0; j < cfgRef->batch; j++)
	{
		int srcShift = j * layerRef[layerIndex - 1].outMat.data.cols;
		int dstShift = j * layerRef[layerIndex].outMat.data.cols;

		float* srcPtr = &layerRef[layerIndex - 1].outMat.data.mat[srcShift];
		float* dstPtr = &layerRef[layerIndex].outMat.data.mat[dstShift];

		cnn_afunc_list[cfgRef->layerCfg[layerIndex].aFunc.id](dstPtr,
				srcPtr, layerRef[layerIndex].outMat.data.cols, NULL);
	}
}

inline void cnn_forward_conv(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef,
		int layerIndex)
{
	for(int j = 0; j < cfgRef->batch; j++)
	{
		int srcShift = j * layerRef[layerIndex - 1].outMat.data.cols;
		int dstShift = j * layerRef[layerIndex].outMat.data.cols;

		float* srcPtr = &layerRef[layerIndex - 1].outMat.data.mat[srcShift];
		float* dstPtr = &layerRef[layerIndex].outMat.data.mat[dstShift];

		// Convolution
		cnn_conv_2d(dstPtr,
				layerRef[layerIndex].outMat.height, layerRef[layerIndex].outMat.width,
				layerRef[layerIndex].conv.kernel.mat, cfgRef->layerCfg[layerIndex].conv.size,
				layerRef[layerIndex].conv.inChannel,
				srcPtr, layerRef[layerIndex - 1].outMat.height,
				layerRef[layerIndex - 1].outMat.width);

		// Add bias
		cblas_saxpy(layerRef[layerIndex].conv.bias.cols, 1.0,
				layerRef[layerIndex].conv.bias.mat, 1,
				&layerRef[layerIndex].outMat.data.mat[dstShift], 1);
	}
}

inline void cnn_forward_pool(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef,
		int layerIndex)
{
	for(int j = 0; j < cfgRef->batch; j++)
	{
		int srcShift = j * layerRef[layerIndex - 1].outMat.data.cols;
		int dstShift = j * layerRef[layerIndex].outMat.data.cols;

		float* srcPtr = &layerRef[layerIndex - 1].outMat.data.mat[srcShift];
		float* dstPtr = &layerRef[layerIndex].outMat.data.mat[dstShift];

		cnn_pool_2d_max(dstPtr, (&layerRef[layerIndex].pool.indexMat[dstShift]),
				layerRef[layerIndex].outMat.height, layerRef[layerIndex].outMat.width,
				cfgRef->layerCfg[layerIndex].pool.size, srcPtr,
				layerRef[layerIndex - 1].outMat.height, layerRef[layerIndex - 1].outMat.width);
	}
}

inline void cnn_bp_fc(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef, int layerIndex,
		float lRate)
{
	int srcShift;

	// Find weight delta matrix
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
			layerRef[layerIndex].fc.weight.rows,
			layerRef[layerIndex].fc.weight.cols,
			cfgRef->batch,
			1.0,
			layerRef[layerIndex - 1].outMat.data.mat, layerRef[layerIndex - 1].outMat.data.cols,
			layerRef[layerIndex].outMat.data.grad, layerRef[layerIndex].outMat.data.cols, 0.0,
			layerRef[layerIndex].fc.weight.grad, layerRef[layerIndex].fc.weight.cols);

	// Find bias delta matrix
	memset(layerRef[layerIndex].fc.bias.grad, 0, sizeof(float) *
			layerRef[layerIndex].fc.bias.cols);

	for(int j = 0; j < cfgRef->batch; j++)
	{
		srcShift = j * layerRef[layerIndex].outMat.data.cols;
		cblas_saxpy(layerRef[layerIndex].fc.bias.cols, 1.0,
				&layerRef[layerIndex].outMat.data.grad[srcShift],
				1, layerRef[layerIndex].fc.bias.grad, 1);
	}

	// Find layer gradient
	if(layerIndex > 1)
	{
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				layerRef[layerIndex - 1].outMat.data.rows,
				layerRef[layerIndex - 1].outMat.data.cols,
				layerRef[layerIndex].outMat.data.cols,
				1.0,
				layerRef[layerIndex].outMat.data.grad, layerRef[layerIndex].outMat.data.cols,
				layerRef[layerIndex].fc.weight.mat, layerRef[layerIndex].fc.weight.cols, 0.0,
				layerRef[layerIndex - 1].outMat.data.grad,
				layerRef[layerIndex - 1].outMat.data.cols);
	}

	// Update weight
	cblas_saxpy(layerRef[layerIndex].fc.weight.rows * layerRef[layerIndex].fc.weight.cols,
			lRate,
			layerRef[layerIndex].fc.weight.grad, 1,
			layerRef[layerIndex].fc.weight.mat, 1);

	// Update bias
	cblas_saxpy(layerRef[layerIndex].fc.bias.cols, lRate,
			layerRef[layerIndex].fc.bias.grad, 1,
			layerRef[layerIndex].fc.bias.mat, 1);
}

inline void cnn_bp_drop(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef, int layerIndex,
		float lRate)
{
	if(layerIndex > 1)
	{
		int size = layerRef[layerIndex].outMat.data.cols;
		int* mask = layerRef[layerIndex].drop.mask;

		// Zero gradient memory
		memset(layerRef[layerIndex - 1].outMat.data.grad, 0, sizeof(float) *
				layerRef[layerIndex].outMat.data.rows *
				layerRef[layerIndex].outMat.data.cols);

		// Find layer gradient
		for(int j = 0; j < cfgRef->batch; j++)
		{
			int shift = j * layerRef[layerIndex].outMat.data.cols;

			cnn_drop_grad(&layerRef[layerIndex - 1].outMat.data.grad[shift],
					&layerRef[layerIndex].outMat.data.grad[shift],
					mask, size);
		}
	}
}

inline void cnn_bp_afunc(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef, int layerIndex,
		float lRate)
{
	int srcShift, dstShift;
	float* srcPtr;
	float* dstPtr;

	if(layerIndex > 1)
	{
		for(int j = 0; j < cfgRef->batch; j++)
		{
			srcShift = j * layerRef[layerIndex - 1].outMat.data.cols;
			if(cfgRef->layerCfg[layerIndex].aFunc.id == CNN_SOFTMAX)
			{
				dstShift = srcShift * layerRef[layerIndex].outMat.data.cols;
			}
			else
			{
				dstShift = srcShift;
			}

			srcPtr = &layerRef[layerIndex - 1].outMat.data.mat[srcShift];
			dstPtr = &layerRef[layerIndex].aFunc.gradMat.mat[dstShift];

			// Find gradient matrix
			cnn_afunc_grad_list[cfgRef->layerCfg[layerIndex].aFunc.id](dstPtr,
					srcPtr, layerRef[layerIndex].outMat.data.cols,
					&layerRef[layerIndex].aFunc.buf.mat[srcShift]);

			// Find layer gradient
			if(cfgRef->layerCfg[layerIndex].aFunc.id == CNN_SOFTMAX)
			{
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						1, layerRef[layerIndex - 1].outMat.data.cols,
						layerRef[layerIndex].outMat.data.cols,
						1.0,
						&layerRef[layerIndex].outMat.data.grad[srcShift],
						layerRef[layerIndex].outMat.data.cols,
						dstPtr, layerRef[layerIndex].aFunc.gradMat.cols, 0.0,
						&layerRef[layerIndex - 1].outMat.data.grad[srcShift],
						layerRef[layerIndex - 1].outMat.data.cols);
			}
			else
			{
				for(int k = 0; k < layerRef[layerIndex - 1].outMat.data.cols; k++)
				{
					layerRef[layerIndex - 1].outMat.data.grad[srcShift + k] = dstPtr[k] *
						layerRef[layerIndex].outMat.data.grad[srcShift + k];
				}
			}
		}
	}
}

inline void cnn_bp_conv(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef, int layerIndex,
		float lRate)
{
	int j;
	int srcShift, dstShift;

	// Find kernel delta matrix
	memset(layerRef[layerIndex].conv.kernel.grad, 0, sizeof(float) *
			layerRef[layerIndex].conv.kernel.rows *
			layerRef[layerIndex].conv.kernel.cols);

	for(j = 0; j < cfgRef->batch; j++)
	{
		srcShift = j * layerRef[layerIndex].outMat.data.cols;
		dstShift = j * layerRef[layerIndex - 1].outMat.data.cols;

		cnn_conv_2d_kernel_grad((&layerRef[layerIndex].outMat.data.grad[srcShift]),
				layerRef[layerIndex].outMat.height,
				layerRef[layerIndex].outMat.width,
				layerRef[layerIndex].conv.kernel.grad,
				layerRef[layerIndex].conv.kernel.cols,
				layerRef[layerIndex].conv.inChannel,
				(&layerRef[layerIndex - 1].outMat.data.mat[dstShift]),
				layerRef[layerIndex - 1].outMat.height,
				layerRef[layerIndex - 1].outMat.width);
	}

	// Find bias delta matrix
	memset(layerRef[layerIndex].conv.bias.grad, 0, sizeof(float) *
			layerRef[layerIndex].conv.bias.cols);

	for(j = 0; j < cfgRef->batch; j++)
	{
		srcShift = j * layerRef[layerIndex].outMat.data.cols;
		cblas_saxpy(layerRef[layerIndex].conv.bias.cols, 1.0,
				&layerRef[layerIndex].outMat.data.grad[srcShift],
				1, layerRef[layerIndex].conv.bias.grad, 1);
	}

	// Find layer gradient
	if(layerIndex > 1)
	{
		memset(layerRef[layerIndex - 1].outMat.data.grad, 0, sizeof(float) *
				layerRef[layerIndex - 1].outMat.data.rows *
				layerRef[layerIndex - 1].outMat.data.cols);

		for(j = 0; j < cfgRef->batch; j++)
		{
			srcShift = j * layerRef[layerIndex].outMat.data.cols;
			dstShift = j * layerRef[layerIndex - 1].outMat.data.cols;

			cnn_conv_2d_grad((&layerRef[layerIndex - 1].outMat.data.grad[dstShift]),
					layerRef[layerIndex - 1].outMat.height,
					layerRef[layerIndex - 1].outMat.width,
					layerRef[layerIndex].conv.kernel.mat,
					layerRef[layerIndex].conv.kernel.cols,
					layerRef[layerIndex].conv.inChannel,
					(&layerRef[layerIndex].outMat.data.grad[srcShift]),
					layerRef[layerIndex].outMat.height,
					layerRef[layerIndex].outMat.width);
		}
	}

	// Update kernel
	cblas_saxpy(layerRef[layerIndex].conv.kernel.cols *
			layerRef[layerIndex].conv.kernel.rows, lRate,
			layerRef[layerIndex].conv.kernel.grad, 1,
			layerRef[layerIndex].conv.kernel.mat, 1);

	// Update bias
	cblas_saxpy(layerRef[layerIndex].conv.bias.cols, lRate,
			layerRef[layerIndex].conv.bias.grad, 1,
			layerRef[layerIndex].conv.bias.mat, 1);
}

inline void cnn_bp_pool(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef, int layerIndex,
		float lRate)
{
	int srcShift, dstShift;
	float* srcPtr;

	if(layerIndex > 1)
	{
		// Zero layer gradient
		memset(layerRef[layerIndex - 1].outMat.data.grad, 0, sizeof(float) *
				layerRef[layerIndex - 1].outMat.data.rows *
				layerRef[layerIndex - 1].outMat.data.cols);

		for(int j = 0; j < cfgRef->batch; j++)
		{
			srcShift = j * layerRef[layerIndex].outMat.data.cols;
			dstShift = j * layerRef[layerIndex - 1].outMat.data.cols;

			srcPtr = &layerRef[layerIndex].outMat.data.grad[srcShift];

			// Find layer gradient
			cnn_pool_2d_max_grad((&layerRef[layerIndex - 1].outMat.data.grad[dstShift]),
					(&layerRef[layerIndex].pool.indexMat[srcShift]),
					srcPtr, layerRef[layerIndex].outMat.height,
					layerRef[layerIndex].outMat.width);
		}
	}
}

#endif
