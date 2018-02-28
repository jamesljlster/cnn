#ifndef __CNN_CALC_H__
#define __CNN_CALC_H__

#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#include "cnn.h"
#include "cnn_types.h"
#include "cnn_builtin_math.h"

inline void cnn_drop(float* dst, float* src, int* mask, int size, float scale)
{
	for(int __i = 0; __i < size; __i++)
	{
		if(mask[__i] > 0)
		{
			dst[__i] = src[__i] * scale;
		}
		else
		{
			dst[__i] = 0;
		}
	}
}

inline void cnn_drop_grad(float* gradDst, float* gradSrc, int* mask, int size, float scale)
{
	for(int __i = 0; __i < size; __i++)
	{
		if(mask[__i] > 0)
		{
			gradDst[__i] += gradSrc[__i] * scale;
		}
		else
		{
			gradDst[__i] = 0;
		}
	}
}

inline void cnn_conv_2d(float* dst, int dstHeight, int dstWidth,
		float* kernel, int kSize, int chIn, int chOut,
		float* src, int srcHeight, int srcWidth)
{
	int __kMemSize = kSize * kSize;
	int __filterSize = chIn * __kMemSize;
	int __dstImSize = dstHeight * dstWidth;
	int __srcImSize = srcHeight * srcWidth;

	for(int __chOut = 0; __chOut < chOut; __chOut++)
	{
		int __filterShift = __chOut * __filterSize;
		int __dstChShift = __chOut * __dstImSize;

		for(int __chIn = 0; __chIn < chIn; __chIn++)
		{
			int __kShiftBase = __chIn * __kMemSize + __filterShift;
			int __srcChShift = __chIn * __srcImSize;

			for(int __h = 0; __h < dstHeight; __h++)
			{
				int __dstShift = __h * dstWidth + __dstChShift;
				for(int __w = 0; __w < dstWidth; __w++)
				{
					float __conv = 0;
					for(int __convH = 0; __convH < kSize; __convH++)
					{
						int __kShift = __convH * kSize + __kShiftBase;
						int __srcShift = (__h + __convH) * srcWidth + __srcChShift;

						for(int __convW = 0; __convW < kSize; __convW++)
						{
							__conv += kernel[__kShift + __convW] *
								src[__srcShift + (__w + __convW)];
						}
					}

					dst[__dstShift + __w] += __conv;
				}
			}
		}
	}
}

inline void cnn_conv_2d_grad(float* srcGrad, int srcHeight, int srcWidth,
		float* kernel, int kSize, int srcCh, int lCh,
		float* lGrad, int lHeight, int lWidth)
{
	int __kMemSize = kSize * kSize;
	int __filterSize = srcCh * __kMemSize;
	int __lImSize = lHeight * lWidth;
	int __srcImSize = srcHeight * srcWidth;

	for(int __lCh = 0; __lCh < lCh; __lCh++)
	{
		int __filterShift = __lCh * __filterSize;
		int __lChShift = __lCh * __lImSize;

		for(int __srcCh = 0; __srcCh < srcCh; __srcCh++)
		{
			int __srcChShift = __srcCh * __srcImSize;
			int __kShiftBase = __srcCh * __kMemSize + __filterShift;

			for(int __h = 0; __h < lHeight; __h++)
			{
				int __lShift = __h * lHeight + __lChShift;
				for(int __w = 0; __w < lWidth; __w++)
				{
					for(int __convH = 0; __convH < kSize; __convH++)
					{
						int __kShift = __convH * kSize + __kShiftBase;
						int __srcShift = (__h + __convH) * srcWidth + __srcChShift;

						for(int __convW = 0; __convW < kSize; __convW++)
						{
							srcGrad[__srcShift + (__w + __convW)] +=
								lGrad[__lShift + __w] * kernel[__kShift + __convW];
						}
					}
				}
			}
		}
	}
}

inline void cnn_conv_2d_kernel_grad(float* lGrad, int lHeight, int lWidth,
		float* kGrad, int kSize, int lCh, int srcCh,
		float* src, int srcHeight, int srcWidth)
{
	int __kMemSize = kSize * kSize;
	int __filterSize = srcCh * __kMemSize;
	int __lImSize = lHeight * lWidth;
	int __srcImSize = srcHeight * srcWidth;

	for(int __lCh = 0; __lCh < lCh; __lCh++)
	{
		int __filterShift = __lCh * __filterSize;
		int __lChShift = __lCh * __lImSize;

		for(int __srcCh = 0; __srcCh < srcCh; __srcCh++)
		{
			int __srcChShift = __srcCh * __srcImSize;
			int __kShiftBase = __srcCh * __kMemSize + __filterShift;

			for(int __h = 0; __h < lHeight; __h++)
			{
				int __lShift = __h * lWidth + __lChShift;
				for(int __w = 0; __w < lWidth; __w++)
				{
					for(int __convH = 0; __convH < kSize; __convH++)
					{
						int __kShift = __convH * kSize + __kShiftBase;
						int __srcShift = (__h + __convH) * srcWidth + __srcChShift;

						for(int __convW = 0; __convW < kSize; __convW++)
						{
							kGrad[__kShift + __convW] +=
								lGrad[__lShift + __w] * src[__srcShift + (__w + __convW)];
						}
					}
				}
			}
		}
	}
}

inline void cnn_pool_2d_max(float* dst, int* indexMat, int dstHeight, int dstWidth,
		float* src, int srcWidth, int srcHeight, int poolSize, int channel)
{
	int __dstImSize = dstHeight * dstWidth;
	int __srcImSize = srcHeight * srcWidth;

	for(int __ch = 0; __ch < channel; __ch++)
	{
		int __dstChShift = __ch * __dstImSize;
		int __srcChShift = __ch * __srcImSize;

		for(int __h = 0; __h < dstHeight; __h++)
		{
			for(int __w = 0; __w < dstWidth; __w++)
			{
				float __tmp, __max;
				int __maxIndex, __index;

				__index = (__h * poolSize) * srcWidth + (__w * poolSize) + __srcChShift;
				__max = src[__index];
				__maxIndex = __index;
				for(int __poolH = 0; __poolH < poolSize; __poolH++)
				{
					for(int __poolW = 0; __poolW < poolSize; __poolW++)
					{
						__index = ((__h * poolSize) + __poolH) * srcWidth +
							((__w * poolSize) + __poolW) + __srcChShift;
						__tmp = src[__index];
						if(__tmp > __max)
						{
							__max = __tmp;
							__maxIndex = __index;
						}
					}
				}

				__index = __h * dstWidth + __w + __dstChShift;
				dst[__index] = __max;
				indexMat[__index] = __maxIndex;
			}
		}
	}
}

inline void cnn_pool_2d_max_grad(float* grad, int* indexMat,
		float* iGrad, int iGradRows, int iGradCols, int iCh)
{
	int size = iGradRows * iGradCols * iCh;
	for(int __i = 0; __i < size; __i++)
	{
		grad[indexMat[__i]] += iGrad[__i];
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
	int size = layerRef[layerIndex].outMat.data.rows *
		layerRef[layerIndex].outMat.data.cols;
	int* mask = layerRef[layerIndex].drop.mask;
	float rate = cfgRef->layerCfg[layerIndex].drop.rate;

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

	cnn_drop(layerRef[layerIndex].outMat.data.mat,
			layerRef[layerIndex - 1].outMat.data.mat,
			mask, size, cfgRef->layerCfg[layerIndex].drop.scale);
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
	// Clear outputs
	memset(layerRef[layerIndex].outMat.data.mat, 0, sizeof(float) *
			layerRef[layerIndex].outMat.data.rows *
			layerRef[layerIndex].outMat.data.cols);

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
				cfgRef->layerCfg[layerIndex].conv.filter,
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
	// Clear outputs
	memset(layerRef[layerIndex].outMat.data.mat, 0, sizeof(float) *
			layerRef[layerIndex].outMat.data.rows *
			layerRef[layerIndex].outMat.data.cols);

	for(int j = 0; j < cfgRef->batch; j++)
	{
		int srcShift = j * layerRef[layerIndex - 1].outMat.data.cols;
		int dstShift = j * layerRef[layerIndex].outMat.data.cols;

		float* srcPtr = &layerRef[layerIndex - 1].outMat.data.mat[srcShift];
		float* dstPtr = &layerRef[layerIndex].outMat.data.mat[dstShift];

		cnn_pool_2d_max(dstPtr, &layerRef[layerIndex].pool.indexMat[dstShift],
				layerRef[layerIndex].outMat.height, layerRef[layerIndex].outMat.width,
				srcPtr,
				layerRef[layerIndex - 1].outMat.height, layerRef[layerIndex - 1].outMat.width,
				cfgRef->layerCfg[layerIndex].pool.size,
				layerRef[layerIndex].outMat.channel);
	}
}

inline void cnn_backward_fc(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef,
		int layerIndex)
{
	int srcShift;

	// Sum weight gradient matrix
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
			layerRef[layerIndex].fc.weight.rows,
			layerRef[layerIndex].fc.weight.cols,
			cfgRef->batch,
			1.0,
			layerRef[layerIndex - 1].outMat.data.mat, layerRef[layerIndex - 1].outMat.data.cols,
			layerRef[layerIndex].outMat.data.grad, layerRef[layerIndex].outMat.data.cols, 1.0,
			layerRef[layerIndex].fc.weight.grad, layerRef[layerIndex].fc.weight.cols);

	// Sum bias gradient matrix
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
}

inline void cnn_backward_drop(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef,
		int layerIndex)
{
	if(layerIndex > 1)
	{
		int size = layerRef[layerIndex].outMat.data.rows *
			layerRef[layerIndex].outMat.data.cols;
		int* mask = layerRef[layerIndex].drop.mask;

		// Find layer gradient
		cnn_drop_grad(layerRef[layerIndex - 1].outMat.data.grad,
				layerRef[layerIndex].outMat.data.grad,
				mask, size, cfgRef->layerCfg[layerIndex].drop.scale);
	}
}

inline void cnn_backward_afunc(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef,
		int layerIndex)
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

inline void cnn_backward_conv(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef,
		int layerIndex)
{
	int j;
	int srcShift, dstShift;

	// Sum kernel gradient matrix
	for(j = 0; j < cfgRef->batch; j++)
	{
		srcShift = j * layerRef[layerIndex].outMat.data.cols;
		dstShift = j * layerRef[layerIndex - 1].outMat.data.cols;

		cnn_conv_2d_kernel_grad(&layerRef[layerIndex].outMat.data.grad[srcShift],
				layerRef[layerIndex].outMat.height,
				layerRef[layerIndex].outMat.width,
				layerRef[layerIndex].conv.kernel.grad,
				layerRef[layerIndex].conv.kernel.cols,
				layerRef[layerIndex].outMat.channel,
				layerRef[layerIndex - 1].outMat.channel,
				&layerRef[layerIndex - 1].outMat.data.mat[dstShift],
				layerRef[layerIndex - 1].outMat.height,
				layerRef[layerIndex - 1].outMat.width);
	}

	// Find bias gradient matrix
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
					layerRef[layerIndex - 1].outMat.channel,
					layerRef[layerIndex].outMat.channel,
					&layerRef[layerIndex].outMat.data.grad[srcShift],
					layerRef[layerIndex].outMat.height,
					layerRef[layerIndex].outMat.width);
		}
	}
}

inline void cnn_backward_pool(union CNN_LAYER* layerRef, struct CNN_CONFIG* cfgRef,
		int layerIndex)
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
			cnn_pool_2d_max_grad(&layerRef[layerIndex - 1].outMat.data.grad[dstShift],
					&layerRef[layerIndex].pool.indexMat[srcShift],
					srcPtr, layerRef[layerIndex].outMat.height,
					layerRef[layerIndex].outMat.width,
					layerRef[layerIndex].outMat.channel);
		}
	}
}

#endif
