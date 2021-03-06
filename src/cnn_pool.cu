#include "cnn_cudef.h"

#ifdef __cplusplus
extern "C"
{
#endif

    __global__ void cnn_pool_2d_max_kernel(float* dst, int* indexMat,
                                           int dstWidth, int dstHeight,
                                           int poolSize, float* src,
                                           int srcWidth, int srcHeight,
                                           int channel)
    {
        int w = blockIdx.x * blockDim.x + threadIdx.x;
        int h = blockIdx.y * blockDim.y + threadIdx.y;
        int c = blockIdx.z * blockDim.z + threadIdx.z;
        if (w >= dstWidth || h >= dstHeight || c >= channel)
        {
            return;
        }

        int srcImSize = srcWidth * srcHeight;
        int dstImSize = dstWidth * dstHeight;

        int srcChShift = c * srcImSize;
        int dstChShift = c * dstImSize;

        int hBase = h * poolSize;
        int wBase = w * poolSize;

        float tmp, max;
        int maxIndex, index;

        index = hBase * srcWidth + wBase + srcChShift;
        max = src[index];
        maxIndex = index;

        for (int poolH = 0; poolH < poolSize; poolH++)
        {
            int hShift = hBase + poolH;
            for (int poolW = 0; poolW < poolSize; poolW++)
            {
                int wShift = wBase + poolW;
                index = hShift * srcWidth + wShift + srcChShift;
                tmp = src[index];
                if (tmp > max)
                {
                    max = tmp;
                    maxIndex = index;
                }
            }
        }

        index = h * dstWidth + w + dstChShift;
        dst[index] = max;
        indexMat[index] = maxIndex;
    }

    void cnn_pool_2d_max_gpu(float* dst, int* indexMat, int dstWidth,
                             int dstHeight, int poolSize, float* src,
                             int srcWidth, int srcHeight, int channel)
    {
        int wBlocks = dstWidth / CNN_THREAD_PER_BLOCK_3D;
        if (dstWidth % CNN_THREAD_PER_BLOCK_3D)
        {
            wBlocks += 1;
        }

        int hBlocks = dstHeight / CNN_THREAD_PER_BLOCK_3D;
        if (dstHeight % CNN_THREAD_PER_BLOCK_3D)
        {
            hBlocks += 1;
        }

        int cBlocks = channel / CNN_THREAD_PER_BLOCK_3D;
        if (channel % CNN_THREAD_PER_BLOCK_3D)
        {
            cBlocks += 1;
        }

        dim3 blk(CNN_THREAD_PER_BLOCK_3D, CNN_THREAD_PER_BLOCK_3D,
                 CNN_THREAD_PER_BLOCK_3D);
        dim3 grid(wBlocks, hBlocks, cBlocks);

        cnn_pool_2d_max_kernel<<<grid, blk>>>(dst, indexMat, dstWidth,
                                              dstHeight, poolSize, src,
                                              srcWidth, srcHeight, channel);
    }

    __global__ void cnn_pool_2d_max_grad_kernel(float* grad, int* indexMat,
                                                float* gradIn, int size)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= size)
        {
            return;
        }

        grad[indexMat[index]] = gradIn[index];
    }

    void cnn_pool_2d_max_grad_gpu(float* grad, int* indexMat, float* gradIn,
                                  int size)
    {
        int blocks = size / CNN_THREAD_PER_BLOCK;
        if (size % CNN_THREAD_PER_BLOCK)
        {
            blocks += 1;
        }

        cnn_pool_2d_max_grad_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(
            grad, indexMat, gradIn, size);
    }

#ifdef __cplusplus
}
#endif
