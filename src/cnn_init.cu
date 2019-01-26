#include "cnn_calc.h"
#include "cnn_init_cu.h"

__global__ void cnn_conv_unroll_2d_valid_cu(int* indexMap, int dstHeight,
                                            int dstWidth, int kSize,
                                            int srcHeight, int srcWidth,
                                            int srcCh)
{
    int __kMemSize = kSize * kSize;
    int __srcImSize = srcHeight * srcWidth;
    int __indexMapCols = __kMemSize * srcCh;

    for (int __h = 0; __h < dstHeight; __h++)
    {
        int __dstRowShift = __h * dstHeight;

        for (int __w = 0; __w < dstWidth; __w++)
        {
            int __indexMapRow = __dstRowShift + __w;
            int __indexMemBase = __indexMapRow * __indexMapCols;

            for (int __ch = 0; __ch < srcCh; __ch++)
            {
                int __indexMemShiftBase = __indexMemBase + __kMemSize * __ch;
                int __srcChShift = __ch * __srcImSize;

                for (int __convH = 0; __convH < kSize; __convH++)
                {
                    int __indexMemShift = __indexMemShiftBase + __convH * kSize;
                    int __srcShift = (__h + __convH) * srcWidth + __srcChShift;

                    for (int __convW = 0; __convW < kSize; __convW++)
                    {
                        indexMap[__indexMemShift + __convW] =
                            __srcShift + (__w + __convW);
                    }
                }
            }
        }
    }
}

__global__ void cnn_conv_unroll_2d_same_cu(int* indexMap, int dstHeight,
                                           int dstWidth, int kSize,
                                           int srcHeight, int srcWidth,
                                           int srcCh)
{
    int __kMemSize = kSize * kSize;
    int __srcImSize = srcHeight * srcWidth;
    int __indexMapCols = __kMemSize * srcCh;

    int __convHBase = -kSize / 2;
    int __convWBase = -kSize / 2;

    for (int __i = 0; __i < dstWidth * dstHeight * kSize; __i++)
    {
        indexMap[__i] = -1;
    }

    for (int __h = 0; __h < dstHeight; __h++)
    {
        int __dstRowShift = __h * dstHeight;

        for (int __w = 0; __w < dstWidth; __w++)
        {
            int __indexMapRow = __dstRowShift + __w;
            int __indexMemBase = __indexMapRow * __indexMapCols;

            for (int __ch = 0; __ch < srcCh; __ch++)
            {
                int __indexMemShiftBase = __indexMemBase + __kMemSize * __ch;
                int __srcChShift = __ch * __srcImSize;

                for (int __convH = 0; __convH < kSize; __convH++)
                {
                    int __indexMemShift = __indexMemShiftBase + __convH * kSize;
                    int __convHIndex = __h + __convH + __convHBase;

                    if (__convHIndex >= 0 && __convHIndex < srcHeight)
                    {
                        int __srcShift = __convHIndex * srcWidth + __srcChShift;

                        for (int __convW = 0; __convW < kSize; __convW++)
                        {
                            int __convWIndex = __w + __convW + __convWBase;
                            if (__convWIndex >= 0 && __convWIndex < srcWidth)
                            {
                                int __tmpIndex = __srcShift + __convWIndex;

                                indexMap[__indexMemShift + __convW] =
                                    __tmpIndex;
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void cnn_bn_init_cu(float* bnVar, int ch, float rInit, float bInit)
{
    // Set initial gamma, beta
    for (int i = 0; i < ch; i++)
    {
        bnVar[i * 2 + 0] = rInit;
        bnVar[i * 2 + 1] = bInit;
    }
}
