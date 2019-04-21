#ifndef _DEBUG_H_
#define _DEBUG_H_

#ifdef DEBUG
#include <stdio.h>
#define LOG(msg, ...)                        \
    fprintf(stderr, "%s(): ", __FUNCTION__); \
    fprintf(stderr, msg, ##__VA_ARGS__);     \
    fprintf(stderr, "\n");

#define LOG_MAT(msg, src, rows, cols, ...)                \
    fprintf(stderr, "%s(): ", __FUNCTION__);              \
    fprintf(stderr, msg, ##__VA_ARGS__);                  \
    fprintf(stderr, "\n");                                \
    fprintf(stderr, "[\n");                               \
    for (int _i = 0; _i < rows; _i++)                     \
    {                                                     \
        fprintf(stderr, "[ ");                            \
        for (int _j = 0; _j < cols; _j++)                 \
        {                                                 \
            fprintf(stderr, "%+5e", src[_i * cols + _j]); \
            if (_j < cols - 1)                            \
            {                                             \
                fprintf(stderr, ", ");                    \
            }                                             \
            else                                          \
            {                                             \
                if (_i < rows - 1)                        \
                {                                         \
                    fprintf(stderr, "],\n");              \
                }                                         \
                else                                      \
                {                                         \
                    fprintf(stderr, "]\n");               \
                }                                         \
            }                                             \
        }                                                 \
    }                                                     \
    fprintf(stderr, "]\n");

#define CU_DUMP_L(msg, ptr, len)                                              \
    {                                                                         \
        int* dumpTmp = (int*)calloc(len, sizeof(int));                        \
        if (dumpTmp == NULL)                                                  \
        {                                                                     \
            fprintf(stderr,                                                   \
                    "%s(): Memory allocation failed while trying to dump %s " \
                    "with length %d\n",                                       \
                    __FUNCTION__, #ptr, len);                                 \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            cudaError_t cuErr = cudaMemcpy(dumpTmp, ptr, len * sizeof(int),   \
                                           cudaMemcpyDeviceToHost);           \
            if (cuErr != cudaSuccess)                                         \
            {                                                                 \
                fprintf(stderr,                                               \
                        "%s(): cudaMemcpy() failed while trying to dump %s "  \
                        "with length %d\n",                                   \
                        __FUNCTION__, #ptr, len);                             \
            }                                                                 \
            else                                                              \
            {                                                                 \
                int i;                                                        \
                fprintf(stderr, "%s(): %s", __FUNCTION__, msg);               \
                for (i = 0; i < len; i++)                                     \
                {                                                             \
                    fprintf(stderr, "%d", dumpTmp[i]);                        \
                    if (i == len - 1)                                         \
                    {                                                         \
                        fprintf(stderr, "\n");                                \
                    }                                                         \
                    else                                                      \
                    {                                                         \
                        fprintf(stderr, ", ");                                \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
        free(dumpTmp);                                                        \
    }

#define CU_DUMP_F(msg, ptr, len)                                              \
    {                                                                         \
        float* dumpTmp = (float*)calloc(len, sizeof(float));                  \
        if (dumpTmp == NULL)                                                  \
        {                                                                     \
            fprintf(stderr,                                                   \
                    "%s(): Memory allocation failed while trying to dump %s " \
                    "with length %d\n",                                       \
                    __FUNCTION__, #ptr, len);                                 \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            cudaError_t cuErr = cudaMemcpy(dumpTmp, ptr, len * sizeof(float), \
                                           cudaMemcpyDeviceToHost);           \
            if (cuErr != cudaSuccess)                                         \
            {                                                                 \
                fprintf(stderr,                                               \
                        "%s(): cudaMemcpy() failed while trying to dump %s "  \
                        "with length %d\n",                                   \
                        __FUNCTION__, #ptr, len);                             \
            }                                                                 \
            else                                                              \
            {                                                                 \
                int i;                                                        \
                fprintf(stderr, "%s(): %s", __FUNCTION__, msg);               \
                for (i = 0; i < len; i++)                                     \
                {                                                             \
                    fprintf(stderr, "%f", dumpTmp[i]);                        \
                    if (i == len - 1)                                         \
                    {                                                         \
                        fprintf(stderr, "\n");                                \
                    }                                                         \
                    else                                                      \
                    {                                                         \
                        fprintf(stderr, ", ");                                \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
        free(dumpTmp);                                                        \
    }

#else
#define LOG(msg, ...)
#define LOG_MAT(msg, src, rows, cols, ...)
#define CU_DUMP_L(msg, ptr, len)
#define CU_DUMP_F(msg, ptr, len)
#endif

#endif
