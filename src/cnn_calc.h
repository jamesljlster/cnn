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

#endif
