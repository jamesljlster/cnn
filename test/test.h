#ifndef __TEST_H__
#define __TEST_H__

#define test(func)                                                  \
    {                                                               \
        int __retVal = func;                                        \
        if (__retVal != CNN_NO_ERROR)                               \
        {                                                           \
            fprintf(stderr, "%s(), %d: %s failed with error: %d\n", \
                    __FUNCTION__, __LINE__, #func, __retVal);       \
            return -1;                                              \
        }                                                           \
    }

#define alloc(ptr, size, type)                                              \
    ptr = calloc(size, sizeof(type));                                       \
    if (ptr == NULL)                                                        \
    {                                                                       \
        fprintf(                                                            \
            stderr,                                                         \
            "%s(), %d: Memory allocation failed with size: %d, type: %s\n", \
            __FUNCTION__, __LINE__, size, #type);                           \
        return -1;                                                          \
    }

#define cu_alloc(ptr, size, type)                                             \
    {                                                                         \
        cudaError_t cuRet = cudaMalloc((void**)&ptr, size * sizeof(type));    \
        if (cuRet != cudaSuccess)                                             \
        {                                                                     \
            fprintf(stderr,                                                   \
                    "%s(), %d: Cuda memory allocation failed with size: %d, " \
                    "type: %s\n",                                             \
                    __FUNCTION__, __LINE__, size, #type);                     \
            return -1;                                                        \
        }                                                                     \
    }

#endif
