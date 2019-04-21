#ifndef __CNN_MACRO_H__
#define __CNN_MACRO_H__

#include "cnn_config.h"

// Includes
#ifdef CNN_WITH_CUDA
#include <cuda_runtime.h>
#endif

#ifdef DEBUG
#include <stdio.h>
#endif

// Allocation macros
#ifdef DEBUG
#define cnn_free(ptr)                                                 \
    fprintf(stderr, "%s(): free(%s), %p\n", __FUNCTION__, #ptr, ptr); \
    free(ptr)
#else
#define cnn_free(ptr) free(ptr)
#endif

#define cnn_alloc(ptr, len, type, retVar, errLabel) \
    ptr = calloc(len, sizeof(type));                \
    if (ptr == NULL)                                \
    {                                               \
        retVar = CNN_MEM_FAILED;                    \
        goto errLabel;                              \
    }

// CUDA allocation macros
#ifdef CNN_WITH_CUDA

#ifdef DEBUG
#define cnn_free_cu(ptr)                                                  \
    fprintf(stderr, "%s(): cudaFree(%s), %p\n", __FUNCTION__, #ptr, ptr); \
    cudaFree(ptr)
#else
#define cnn_free_cu(ptr) cudaFree(ptr)
#endif

#ifdef DEBUG
#define cnn_alloc_cu(ptr, len, type, retVar, errLabel)                      \
    {                                                                       \
        cudaError_t cuRet = cudaMalloc((void**)&ptr, len * sizeof(type));   \
        if (cuRet != cudaSuccess)                                           \
        {                                                                   \
            fprintf(stderr,                                                 \
                    "%s(): cudaMalloc(&%s, %lu) failed with error: %d\n",   \
                    __FUNCTION__, #ptr, len * sizeof(type), cuRet);         \
            retVar = CNN_MEM_FAILED;                                        \
            goto errLabel;                                                  \
        }                                                                   \
        cuRet = cudaMemset(ptr, 0, len * sizeof(type));                     \
        if (cuRet != cudaSuccess)                                           \
        {                                                                   \
            fprintf(stderr,                                                 \
                    "%s(): cudaMemset(%s, 0, %lu) failed with error: %d\n", \
                    __FUNCTION__, #ptr, len * sizeof(type), cuRet);         \
            retVar = CNN_MEM_FAILED;                                        \
            goto errLabel;                                                  \
        }                                                                   \
    }
#else
#define cnn_alloc_cu(ptr, len, type, retVar, errLabel)               \
    if (cudaMalloc((void**)&ptr, len * sizeof(type)) != cudaSuccess) \
    {                                                                \
        retVar = CNN_MEM_FAILED;                                     \
        goto errLabel;                                               \
    }                                                                \
    if (cudaMemset(ptr, 0, len * sizeof(type)) != cudaSuccess)       \
    {                                                                \
        retVar = CNN_MEM_FAILED;                                     \
        goto errLabel;                                               \
    }
#endif

#endif

// Error handling macros
#ifdef DEBUG
#define cnn_run(func, retVal, errLabel)                                       \
    retVal = func;                                                            \
    if (retVal != CNN_NO_ERROR)                                               \
    {                                                                         \
        fprintf(stderr, "%s(), %d: %s failed with error: %d\n", __FUNCTION__, \
                __LINE__, #func, retVal);                                     \
        goto errLabel;                                                        \
    }
#else
#define cnn_run(func, retVal, errLabel) \
    retVal = func;                      \
    if (retVal != CNN_NO_ERROR)         \
    {                                   \
        goto errLabel;                  \
    }
#endif

// Cuda error handling macros
#ifdef DEBUG
#define cnn_run_cu(func, retVal, errLabel)                          \
    {                                                               \
        cudaError_t cuRet = func;                                   \
        if (cuRet != cudaSuccess)                                   \
        {                                                           \
            fprintf(stderr, "%s(), %d: %s failed with error: %d\n", \
                    __FUNCTION__, __LINE__, #func, cuRet);          \
            retVal = CNN_CUDA_RUNTIME_ERROR;                        \
            goto errLabel;                                          \
        }                                                           \
    }
#else
#define cnn_run_cu(func, retVal, errLabel)   \
    {                                        \
        if (func != cudaSuccess)             \
        {                                    \
            retVal = CNN_CUDA_RUNTIME_ERROR; \
            goto errLabel;                   \
        }                                    \
    }
#endif

#endif
