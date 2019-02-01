#include <cuda_runtime.h>

#include "cnn_builtin_math_cu.h"
#include "cnn_cudef.h"

__device__ float max_cu(float src1, float src2)
{
    return (src1 > src2) ? src1 : src2;
}

__global__ void cnn_max_kernel(float* vec, int len, int slice, int stride,
                               int shift)
{
    int sliceIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (sliceIndex >= slice)
    {
        return;
    }

    int head = sliceIndex * stride;
    int cmp = head + shift;
    if (cmp < len)
    {
        vec[head] = max_cu(vec[head], vec[cmp]);
    }
}

void cnn_max_gpu(float* maxPtr, float* vec, int len, float* buf)
{
    // Copy memory
    cudaMemcpy(buf, vec, len * sizeof(float), cudaMemcpyDeviceToDevice);

    // Find max
    int stride = 2;
    int shift = 1;
    int slice = len;

    while (1)
    {
        // Find slice
        int tmp = slice / 2;
        if (slice % 2)
        {
            tmp = tmp + 1;
        }

        slice = tmp;

        // Run kernel
        int blocks = slice / CNN_THREAD_PER_BLOCK;
        if (slice % CNN_THREAD_PER_BLOCK)
        {
            blocks += 1;
        }
        cnn_max_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(buf, len, slice,
                                                         stride, shift);

        if (slice == 1)
        {
            break;
        }

        // Find new stride, shift
        stride *= 2;
        shift *= 2;
    }

    // Copy result
    cudaMemcpy(maxPtr, buf, sizeof(float), cudaMemcpyDeviceToHost);
}

__device__ float add_cu(float src1, float src2) { return src1 + src2; }

__global__ void cnn_sum_kernel(float* vec, int len, int slice, int stride,
                               int shift)
{
    int sliceIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (sliceIndex >= slice)
    {
        return;
    }

    int head = sliceIndex * stride;
    int cmp = head + shift;
    if (cmp < len)
    {
        vec[head] = add_cu(vec[head], vec[cmp]);
    }
}

void cnn_sum_gpu(float* sumPtr, float* vec, int len, float* buf)
{
    // Copy memory
    cudaMemcpy(buf, vec, len * sizeof(float), cudaMemcpyDeviceToDevice);

    // Find sum
    int stride = 2;
    int shift = 1;
    int slice = len;

    while (1)
    {
        // Find slice
        int tmp = slice / 2;
        if (slice % 2)
        {
            tmp = tmp + 1;
        }

        slice = tmp;

        // Run kernel
        int blocks = slice / CNN_THREAD_PER_BLOCK;
        if (slice % CNN_THREAD_PER_BLOCK)
        {
            blocks += 1;
        }
        cnn_sum_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(buf, len, slice,
                                                         stride, shift);

        if (slice == 1)
        {
            break;
        }

        // Find new stride, shift
        stride *= 2;
        shift *= 2;
    }

    // Copy result
    cudaMemcpy(sumPtr, buf, sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void cnn_add_kernel(float* dst, float* src, int len, float addend)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len)
    {
        return;
    }

    dst[index] = src[index] + addend;
}

void cnn_add_gpu(float* dst, float* src, int len, float addend)
{
    int blocks = len / CNN_THREAD_PER_BLOCK;
    if (len % CNN_THREAD_PER_BLOCK)
    {
        blocks += 1;
    }

    cnn_add_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(dst, src, len, addend);
}

__global__ void cnn_exp_kernel(float* dst, float* src, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len)
    {
        return;
    }

    dst[index] = __expf(src[index]);
}

void cnn_exp_gpu(float* dst, float* src, int len)
{
    int blocks = len / CNN_THREAD_PER_BLOCK;
    if (len % CNN_THREAD_PER_BLOCK)
    {
        blocks += 1;
    }

    cnn_exp_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(dst, src, len);
}

__global__ void cnn_div_kernel(float* dst, float* src, int len, float divider)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len)
    {
        return;
    }

    dst[index] = src[index] / divider;
}

void cnn_div_gpu(float* dst, float* src, int len, float divider)
{
    int blocks = len / CNN_THREAD_PER_BLOCK;
    if (len % CNN_THREAD_PER_BLOCK)
    {
        blocks += 1;
    }

    cnn_div_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(dst, src, len, divider);
}

__global__ void cnn_fmaxf_kernel(float* dst, float* src, int len, float num)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len)
    {
        return;
    }

    dst[index] = fmaxf(src[index], num);
}

void cnn_fmaxf_gpu(float* dst, float* src, int len, float num)
{
    int blocks = len / CNN_THREAD_PER_BLOCK;
    if (len % CNN_THREAD_PER_BLOCK)
    {
        blocks += 1;
    }

    cnn_fmaxf_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(dst, src, len, num);
}

__global__ void cnn_smax_grad_kernel(float* dst, float* cache, int len)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len || j >= len)
    {
        return;
    }

    dst[i * len + j] = cache[i] * ((float)(i == j) - cache[j]);
}

void cnn_smax_grad_gpu(float* dst, float* cache, int len)
{
    int blocks = len / CNN_THREAD_PER_BLOCK_2D;
    if (len % CNN_THREAD_PER_BLOCK_2D)
    {
        blocks += 1;
    }

    dim3 blk(CNN_THREAD_PER_BLOCK_2D, CNN_THREAD_PER_BLOCK_2D);
    dim3 grid(blocks, blocks);

    cnn_smax_grad_kernel<<<grid, blk>>>(dst, cache, len);
}

__global__ void cnn_relu_grad_kernel(float* dst, float* src, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len)
    {
        return;
    }

    dst[index] = (src[index] < 0.0f) ? 0 : 1;
}

void cnn_relu_grad_gpu(float* dst, float* src, int len)
{
    int blocks = len / CNN_THREAD_PER_BLOCK;
    if (len % CNN_THREAD_PER_BLOCK)
    {
        blocks += 1;
    }

    cnn_relu_grad_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(dst, src, len);
}
