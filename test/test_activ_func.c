#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cnn.h>
#include <cnn_builtin_math.h>
#include <cnn_config.h>

#include <debug.h>
#include "test.h"

#ifdef CNN_WITH_CUDA
#include <cuda_runtime.h>
#endif

int main(int argc, char* argv[])
{
    int id;
    int i, j;
    int len;

    float* src = NULL;
    float* dst = NULL;
    float* buf = NULL;
    float* deri = NULL;
    float* grad = NULL;

#ifdef CNN_WITH_CUDA
    float* cuSrc = NULL;
    float* cuDst = NULL;
    float* cuBuf = NULL;
    float* cuDeri = NULL;
    float* cuGrad = NULL;
#endif

    float err;
    float dx = pow(10, -4);

    // Check dx
    assert(dx != 0.0f);

    // Check argument
    if (argc <= 1)
    {
        printf("Assign arguments with real numbers to run the program\n");
        return -1;
    }

    // Memory allocation
    len = argc - 1;

    alloc(src, len, float);
    alloc(dst, len, float);
    alloc(buf, len, float);
    alloc(deri, len * len, float);
    alloc(grad, len * len, float);

#ifdef CNN_WITH_CUDA
    cu_alloc(cuSrc, len, float);
    cu_alloc(cuDst, len, float);
    cu_alloc(cuBuf, len, float);
    cu_alloc(cuDeri, len * len, float);
    cu_alloc(cuGrad, len * len, float);
#endif

    // Test activation functions
    for (id = 0; id < CNN_ACTIV_AMOUNT; id++)
    {
        // Parse argument
        for (i = 0; i < len; i++)
        {
            src[i] = atof(argv[i + 1]);
        }

        // Find grad
        memset(grad, 0, len * len * sizeof(float));
#ifdef CNN_WITH_CUDA
        cudaMemcpy(cuSrc, src, len * sizeof(float), cudaMemcpyHostToDevice);
        cnn_activ_list[id](cuDst, cuSrc, len, cuBuf);
        cudaMemcpy(dst, cuDst, len * sizeof(float), cudaMemcpyDeviceToHost);
#else
        cnn_activ_list[id](dst, src, len, NULL);
#endif
        if (id == CNN_SOFTMAX)
        {
            for (i = 0; i < len; i++)
            {
                for (j = 0; j < len; j++)
                {
                    src[j] = atof(argv[j + 1]);
                    if (i == j)
                    {
                        src[j] += dx;
                    }
                }

#ifdef CNN_WITH_CUDA
                cudaMemcpy(cuSrc, src, len * sizeof(float),
                           cudaMemcpyHostToDevice);
                cnn_activ_list[id](cuBuf, cuSrc, len, cuDeri);
                cudaMemcpy(buf, cuBuf, len * sizeof(float),
                           cudaMemcpyDeviceToHost);
#else
                cnn_activ_list[id](buf, src, len, NULL);
#endif

                for (j = 0; j < len; j++)
                {
                    grad[i * len + j] = (buf[j] - dst[j]) / dx;
                }
            }
        }
        else
        {
            for (i = 0; i < len; i++)
            {
                for (j = 0; j < len; j++)
                {
                    src[j] = atof(argv[j + 1]);
                    if (i == j)
                    {
                        src[j] += dx;
                    }
                }

#ifdef CNN_WITH_CUDA
                cudaMemcpy(cuSrc, src, len * sizeof(float),
                           cudaMemcpyHostToDevice);
                cnn_activ_list[id](cuBuf, cuSrc, len, cuDeri);
                cudaMemcpy(buf, cuBuf, len * sizeof(float),
                           cudaMemcpyDeviceToHost);
#else
                cnn_activ_list[id](buf, src, len, NULL);
#endif

                grad[i] = (buf[i] - dst[i]) / dx;
            }
        }

        // Find derivative
        for (i = 0; i < len; i++)
        {
            src[i] = atof(argv[i + 1]);
        }

        memset(deri, 0, len * len * sizeof(float));

#ifdef CNN_WITH_CUDA
        cudaMemset(cuDeri, 0, len * len * sizeof(float));
        cudaMemcpy(cuSrc, src, len * sizeof(float), cudaMemcpyHostToDevice);
        cnn_activ_grad_list[id](cuDeri, cuSrc, len, cuDst);
        cudaMemcpy(deri, cuDeri, len * len * sizeof(float),
                   cudaMemcpyDeviceToHost);
#else
        cnn_activ_grad_list[id](deri, src, len, dst);
#endif

        // Find error
        err = 0;
        for (i = 0; i < len * len; i++)
        {
            err += fabs(grad[i] - deri[i]);
        }

        printf("=== Test %s derivative ===\n", cnn_activ_name[id]);
        printf("deri:\n");
        print_mat(deri, len, len);
        printf("\n");
        printf("grad:\n");
        print_mat(grad, len, len);
        printf("\n");
        printf("Sum of error: %lf\n", err);
        printf("\n");
    }

    return 0;
}
