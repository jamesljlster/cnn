#ifndef __CNN_CALC_H__
#define __CNN_CALC_H__

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

#endif
