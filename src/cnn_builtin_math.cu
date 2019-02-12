#include <cuda_runtime.h>

#include "cnn_builtin_math_cu.h"
#include "cnn_cudef.h"

#define CNN_SCALAR_ACTIV_IMPL(name, fwProc, bpProc)                           \
    __global__ void cnn_##name##_kernel(float* dst, float* src, int len)      \
    {                                                                         \
        int index = blockIdx.x * blockDim.x + threadIdx.x;                    \
        if (index >= len)                                                     \
        {                                                                     \
            return;                                                           \
        }                                                                     \
                                                                              \
        fwProc                                                                \
    }                                                                         \
                                                                              \
    void cnn_##name##_gpu(float* dst, float* src, int len)                    \
    {                                                                         \
        int blocks = len / CNN_THREAD_PER_BLOCK;                              \
        if (len % CNN_THREAD_PER_BLOCK)                                       \
        {                                                                     \
            blocks += 1;                                                      \
        }                                                                     \
                                                                              \
        cnn_##name##_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(dst, src, len); \
    }                                                                         \
                                                                              \
    __global__ void cnn_##name##_grad_kernel(float* dst, float* src,          \
                                             float* cache, int len)           \
    {                                                                         \
        int index = blockIdx.x * blockDim.x + threadIdx.x;                    \
        if (index >= len)                                                     \
        {                                                                     \
            return;                                                           \
        }                                                                     \
                                                                              \
        bpProc                                                                \
    }                                                                         \
                                                                              \
    void cnn_##name##_grad_gpu(float* dst, float* src, float* cache, int len) \
    {                                                                         \
        int blocks = len / CNN_THREAD_PER_BLOCK;                              \
        if (len % CNN_THREAD_PER_BLOCK)                                       \
        {                                                                     \
            blocks += 1;                                                      \
        }                                                                     \
                                                                              \
        cnn_##name##_grad_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(           \
            dst, src, cache, len);                                            \
    }

#ifdef __cplusplus
extern "C"
{
#endif

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

    __global__ void cnn_add_kernel(float* dst, float* src, int len,
                                   float addend)
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

        dst[index] = expf(src[index]);
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

    __global__ void cnn_div_kernel(float* dst, float* src, int len,
                                   float divider)
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

        cnn_div_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(dst, src, len,
                                                         divider);
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

    __global__ void cnn_elemwise_product_kernel(float* dst, float* src1,
                                                float* src2, int len)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len)
        {
            return;
        }

        dst[index] = src1[index] * src2[index];
    }

    void cnn_elemwise_product_gpu(float* dst, float* src1, float* src2, int len)
    {
        int blocks = len / CNN_THREAD_PER_BLOCK;
        if (len % CNN_THREAD_PER_BLOCK)
        {
            blocks += 1;
        }

        cnn_elemwise_product_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(
            dst, src1, src2, len);
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

    CNN_SCALAR_ACTIV_IMPL(                         //
        relu,                                      //
        dst[index] = fmaxf(src[index], 0.0f);      //
        ,                                          //
        dst[index] = (src[index] < 0.0f) ? 0 : 1;  //
    )

    CNN_SCALAR_ACTIV_IMPL(                                     //
        swish,                                                 //
        dst[index] = src[index] / (1.0f + expf(-src[index]));  //
        ,                                                      //
        if (src[index] == 0.0f)                                //
        {                                                      //
            dst[index] = 0.5;                                  //
        }                                                      //
        else                                                   //
        {                                                      //
            dst[index] = cache[index] + (cache[index] / src[index]) *
                                            (1.0f - cache[index]);  //
        }                                                           //
    )

    CNN_SCALAR_ACTIV_IMPL(                                 //
        sigmoid,                                           //
        dst[index] = 1.0f / (1.0f + expf(-src[index]));    //
        ,                                                  //
        dst[index] = cache[index] * (1.0 - cache[index]);  //
    )

    CNN_SCALAR_ACTIV_IMPL(                                        //
        tanh,                                                     //
        dst[index] = 2.0 / (1.0 + exp(-2.0 * src[index])) - 1.0;  //
        ,                                                         //
        dst[index] = 1.0 - cache[index] * cache[index];           //
    )

    CNN_SCALAR_ACTIV_IMPL(                                 //
        gaussian,                                          //
        dst[index] = exp(-src[index] * src[index] * 0.5);  //
        ,                                                  //
        dst[index] = -src[index] * cache[index];           //
    )

    CNN_SCALAR_ACTIV_IMPL(  //
        bent_identity,      //
        dst[index] = (sqrt(src[index] * src[index] + 1.0) - 1.0) / 2.0 +
                     src[index];  //
        ,                         //
        dst[index] = src[index] / (2.0 * sqrt(src[index] * src[index] + 1.0)) +
                     1.0;  //
    )

    CNN_SCALAR_ACTIV_IMPL(                            //
        softplus,                                     //
        dst[index] = log1p(exp(src[index]));          //
        ,                                             //
        dst[index] = 1.0 / (1.0 + exp(-src[index]));  //
    )

    CNN_SCALAR_ACTIV_IMPL(                                   //
        softsign,                                            //
        dst[index] = src[index] / (1.0 + fabs(src[index]));  //
        ,                                                    //
        float tmp = 1.0 + fabs(src[index]);                  //
        dst[index] = 1.0 / (tmp * tmp);                      //
    )

    CNN_SCALAR_ACTIV_IMPL(                              //
        sinc,                                           //
        if (src[index] == 0.0)                          //
        {                                               //
            dst[index] = 1.0;                           //
        }                                               //
        else                                            //
        {                                               //
            dst[index] = sin(src[index]) / src[index];  //
        }                                               //
        ,                                               //
        if (src[index] == 0.0)                          //
        {                                               //
            dst[index] = 0.0;                           //
        }                                               //
        else                                            //
        {                                               //
            dst[index] = (cos(src[index]) / src[index]) -
                         (sin(src[index]) / (src[index] * src[index]));  //
        }                                                                //
    )

    CNN_SCALAR_ACTIV_IMPL(             //
        sin,                           //
        dst[index] = sin(src[index]);  //
        ,                              //
        dst[index] = cos(src[index]);  //
    )

#ifdef __cplusplus
}
#endif
