#include "cnn_cudef.h"

#ifdef __cplusplus
extern "C"
{
#endif

    __global__ void cnn_text_map_kernel(float* dst, float* src, int* map,
                                        int len)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < len)
        {
            dst[index] = src[map[index]];
        }
    }

    void cnn_text_map_gpu(float* dst, float* src, int* map, int len)
    {
        int blocks = len / CNN_THREAD_PER_BLOCK;
        if (len % CNN_THREAD_PER_BLOCK)
        {
            blocks += 1;
        }

        cnn_text_map_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(dst, src, map,
                                                              len);
    }

    __global__ void cnn_text_map_inv_kernel(float* dst, float* src, int* map,
                                            int len)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < len)
        {
            atomicAdd(dst + map[index], src[index]);
        }
    }

    void cnn_text_map_inv_gpu(float* dst, float* src, int* map, int len)
    {
        int blocks = len / CNN_THREAD_PER_BLOCK;
        if (len % CNN_THREAD_PER_BLOCK)
        {
            blocks += 1;
        }

        cnn_text_map_inv_kernel<<<blocks, CNN_THREAD_PER_BLOCK>>>(dst, src, map,
                                                                  len);
    }

    __global__ void cnn_text_find_diff_kernel(float* diffPtr, float* nbrPtr,
                                              float* ctrPtr, int nbrRows,
                                              int nbrCols)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < nbrRows && col < nbrCols)
        {
            int tmpIndex = row * nbrCols + col;
            diffPtr[tmpIndex] = nbrPtr[tmpIndex] - ctrPtr[row];
        }
    }

    void cnn_text_find_diff_gpu(float* diffPtr, float* nbrPtr, float* ctrPtr,
                                int nbrRows, int nbrCols)
    {
        int rowThSize = CNN_THREAD_PER_BLOCK / nbrCols;
        int colThSize = nbrCols;

        int rowBlocks = nbrRows / rowThSize;
        if (nbrRows % rowThSize)
        {
            rowBlocks += 1;
        }

        int colBlocks = 1;

        dim3 blk(colThSize, rowThSize);
        dim3 grid(colBlocks, rowBlocks);

        cnn_text_find_diff_kernel<<<grid, blk>>>(diffPtr, nbrPtr, ctrPtr,
                                                 nbrRows, nbrCols);
    }

    __global__ void cnn_text_find_diff_grad_kernel(float* gradIn,
                                                   float* diffGrad,
                                                   float* nbrGrad,
                                                   float* ctrGrad, int nbrRows,
                                                   int nbrCols)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < nbrRows && col < nbrCols)
        {
            int tmpIndex = row * nbrCols + col;
            float gradTmp = diffGrad[tmpIndex] * gradIn[tmpIndex];
            nbrGrad[tmpIndex] = gradTmp;
            atomicAdd(ctrGrad + row, -gradTmp);
        }
    }

    void cnn_text_find_diff_grad_gpu(float* gradIn, float* diffGrad,
                                     float* nbrGrad, float* ctrGrad,
                                     int nbrRows, int nbrCols)
    {
        int rowThSize = CNN_THREAD_PER_BLOCK / nbrCols;
        int colThSize = nbrCols;

        int rowBlocks = nbrRows / rowThSize;
        if (nbrRows % rowThSize)
        {
            rowBlocks += 1;
        }

        int colBlocks = 1;

        dim3 blk(colThSize, rowThSize);
        dim3 grid(colBlocks, rowBlocks);

        cnn_text_find_diff_grad_kernel<<<grid, blk>>>(
            gradIn, diffGrad, nbrGrad, ctrGrad, nbrRows, nbrCols);
    }

#ifdef __cplusplus
}
#endif
