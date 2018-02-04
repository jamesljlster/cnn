#ifndef __CNN_CALC_H__
#define __CNN_CALC_H__

#define cnn_conv_2d(dst, dstRows, dstCols, kernel, kernelSize, src, srcRows, srcCols) \
	for(int __row = 0; __row < dstRows; __row++) \
	{ \
		for(int __col = 0; __col < dstCols; __col++) \
		{ \
			float __conv = 0; \
			for(int __convRow = 0; __convRow < kernelSize; __convRow++) \
			{ \
				for(int __convCol = 0; __convCol < kernelSize; __convCol++) \
				{ \
					__conv += kernel[__convRow * kernelSize + __convCol] * \
						src[(__row + __convRow) * srcCols + (__col + __convCol)]; \
				} \
			} \
			dst[__row * dstCols + __col] = __conv; \
		} \
	}

#define cnn_conv_2d_grad(grad, gradRows, gradCols, kernel, kSize, iGrad, iGradRows, iGradCols) \
	for(int __row = 0; __row < iGradRows; __row++) \
	{ \
		for(int __col = 0; __col < iGradCols; __col++) \
		{ \
			for(int __convRow = 0; __convRow < kSize; __convRow++) \
			{ \
				for(int __convCol = 0; __convCol < kSize; __convCol++) \
				{ \
					grad[(__row + __convRow) * gradCols + (__col + __convCol)] += \
						kernel[__convRow * kSize + __convCol] * \
						iGrad[__row * iGradCols + __col]; \
				} \
			} \
		} \
	}

#define cnn_conv_2d_kernel_grad(grad, gradRows, gradCols, kGrad, kSize, src, srcRows, srcCols) \
	for(int __row = 0; __row < gradRows; __row++) \
	{ \
		for(int __col = 0; __col < gradCols; __col++) \
		{ \
			for(int __convRow = 0; __convRow < kSize; __convRow++) \
			{ \
				for(int __convCol = 0; __convCol < kSize; __convCol++) \
				{ \
					kGrad[__convRow * kSize + __convCol] += grad[__row * gradCols + __col] * \
						src[(__row + __convRow) * srcCols + (__col + __convCol)]; \
				} \
			} \
		} \
	}

#define cnn_pool_2d_max(dst, indexMat, dstRows, dstCols, poolSize, src, srcRows, srcCols) \
	for(int __row = 0; __row < dstRows; __row++) \
	{ \
		for(int __col = 0; __col < dstCols; __col++) \
		{ \
			float __tmp, __max; \
			int __maxIndex, __index; \
			int __rowShift = __row * poolSize; \
			int __colShift = __col * poolSize; \
			__max = src[__rowShift * srcCols + __colShift]; \
			__maxIndex = __rowShift * srcCols + __colShift; \
			for(int __poolRow = 0; __poolRow < poolSize; __poolRow++) \
			{ \
				for(int __poolCol = 0; __poolCol < poolSize; __poolCol++) \
				{ \
					__index = (__rowShift + __poolRow) * srcCols + (__colShift + __poolCol); \
					__tmp = src[__index]; \
					if(__tmp > __max) \
					{ \
						__max = __tmp; \
						__maxIndex = __index; \
					} \
				} \
			} \
			dst[__row * dstCols + __col] = __max; \
			indexMat[__row * dstCols + __col] = __maxIndex; \
		} \
	}

#define cnn_pool_2d_max_grad(grad, indexMat, iGrad, iGradRows, iGradCols) \
	for(int __i = 0; __i < iGradRows * iGradCols; __i++) \
	{ \
		grad[indexMat[__i]] = iGrad[__i]; \
	}

#endif
