#include "cnn_cudef.h"

#ifdef __cplusplus
extern "C"
{
#endif

    __global__ void cnn_drop_kernel(float* dst, float* src, int* mask, int size,
                                    float scale)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size)
        {
            return;
        }

        if (mask[i] > 0)
        {
            dst[i] = src[i] * scale;
        }
        else
        {
            dst[i] = 0;
        }
    }

    void cnn_drop_gpu(float* dst, float* src, int* mask, int size, float scale)
    {
        int blocks = size / CNN_THREAD_PER_BLOCK;
        if (size % CNN_THREAD_PER_BLOCK)
        {
            blocks += 1;
        }

        cnn_drop_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(dst, src, mask, size,
                                                          scale);
    }

    __global__ void cnn_drop_grad_kernel(float* gradDst, float* gradSrc,
                                         int* mask, int size, float scale)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size)
        {
            return;
        }

        if (mask[i] > 0)
        {
            gradDst[i] = gradSrc[i] * scale;
        }
    }

    void cnn_drop_grad_gpu(float* gradDst, float* gradSrc, int* mask, int size,
                           float scale)
    {
        int blocks = size / CNN_THREAD_PER_BLOCK;
        if (size % CNN_THREAD_PER_BLOCK)
        {
            blocks += 1;
        }

        cnn_drop_grad_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(
            gradDst, gradSrc, mask, size, scale);
    }

    __global__ void cnn_map_kernel(float* dst, float* src, int* map, int len)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len)
        {
            return;
        }

        int tmpIndex = map[index];
        if (tmpIndex >= 0)
        {
            dst[index] = src[tmpIndex];
        }
    }

    void cnn_map_gpu(float* dst, float* src, int* map, int len)
    {
        int blocks = len / CNN_THREAD_PER_BLOCK;
        if (len % CNN_THREAD_PER_BLOCK)
        {
            blocks += 1;
        }

        cnn_map_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(dst, src, map, len);
    }

    __global__ void cnn_map_inv_kernel(float* dst, float* src, int* map,
                                       int len)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len)
        {
            return;
        }

        int tmpIndex = map[index];
        if (tmpIndex >= 0)
        {
            //dst[tmpIndex] += src[index];
            atomicAdd(dst + tmpIndex, src[index]);
        }
    }

    void cnn_map_inv_gpu(float* dst, float* src, int* map, int len)
    {
        int blocks = len / CNN_THREAD_PER_BLOCK;
        if (len % CNN_THREAD_PER_BLOCK)
        {
            blocks += 1;
        }

        cnn_map_inv_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(dst, src, map,
                                                             len);
    }


#ifdef __cplusplus
}
#endif
