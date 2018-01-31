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

#endif
