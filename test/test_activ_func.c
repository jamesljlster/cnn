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

void matmul(float* matA, int rowsA, int colsA, float* matB, int rowsB,
            int colsB, float* matC, int rowsC, int colsC);

int main(int argc, char* argv[])
{
    int id;
    int i, j;
    int len;

    float* src = NULL;
    float* dst = NULL;
    float* buf = NULL;
    float* deri = NULL;
    float* deriBuf = NULL;
    float* gradIn = NULL;
    float* gradOut = NULL;

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
    alloc(buf, len * len, float);
    alloc(deri, len * len, float);
    alloc(deriBuf, len * len, float);
    alloc(gradIn, len, float);
    alloc(gradOut, len, float);

    for (int i = 0; i < len; i++)
    {
        gradIn[i] = 1.0;
    }

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

        // Test derivative and gradient
#ifdef CNN_WITH_CUDA
        cudaMemcpy(cuSrc, src, len * sizeof(float), cudaMemcpyHostToDevice);
        cnn_activ_list[id](cuDst, cuSrc, len, cuBuf);
        cudaMemcpy(dst, cuDst, len * sizeof(float), cudaMemcpyDeviceToHost);
#else
        cnn_activ_list[id](dst, src, len, NULL);
#endif
        if (id == CNN_SOFTMAX)
        {
            // Find derivative matrix
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
                cnn_activ_list[id](deriBuf, src, len, NULL);
#endif

                for (j = 0; j < len; j++)
                {
                    deri[i * len + j] = (deriBuf[j] - dst[j]) / dx;
                }
            }

            // Find gradient
            for (i = 0; i < len; i++)
            {
                src[i] = atof(argv[i + 1]);
            }

#ifdef CNN_WITH_CUDA
            cudaMemset(cuDeri, 0, len * len * sizeof(float));
            cudaMemcpy(cuSrc, src, len * sizeof(float), cudaMemcpyHostToDevice);
            cnn_activ_grad_list[id](cuDeri, cuSrc, len, cuDst);
            cudaMemcpy(deri, cuDeri, len * len * sizeof(float),
                       cudaMemcpyDeviceToHost);
#else
            cnn_activ_grad_list[id](gradOut, gradIn, src, len, dst, buf);
#endif

            // Find error
            err = 0;
            for (i = 0; i < len * len; i++)
            {
                err += fabs(buf[i] - deri[i]);
            }

            printf("=== Test %s derivative ===\n", cnn_activ_name[id]);
            printf("deri:\n");
            print_mat(deri, len, len);
            printf("\n");
            printf("grad:\n");
            print_mat(buf, len, len);
            printf("\n");
            printf("Sum of error: %lf\n", err);
            printf("\n");

            // Find gradient output error
            matmul(deri, len, len, gradIn, len, 1, buf, len, 1);

            err = 0;
            for (i = 0; i < len; i++)
            {
                err += fabs(gradOut[i] - buf[i]);
            }

            printf("Layer gradient with deri:\n");
            print_mat(buf, len, 1);
            printf("\n");
            printf("Layer gradient with grad:\n");
            print_mat(gradOut, len, 1);
            printf("\n");
            printf("Sum of error: %lf\n", err);
            printf("\n");
        }
        else
        {
            // Find derivative vector
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
                cnn_activ_list[id](deriBuf, src, len, NULL);
#endif

                deri[i] = (deriBuf[i] - dst[i]) / dx;
            }

            // Find gradient
            for (i = 0; i < len; i++)
            {
                src[i] = atof(argv[i + 1]);
            }

#ifdef CNN_WITH_CUDA
            cudaMemset(cuDeri, 0, len * len * sizeof(float));
            cudaMemcpy(cuSrc, src, len * sizeof(float), cudaMemcpyHostToDevice);
            cnn_activ_grad_list[id](cuDeri, cuSrc, len, cuDst);
            cudaMemcpy(deri, cuDeri, len * len * sizeof(float),
                       cudaMemcpyDeviceToHost);
#else
            cnn_activ_grad_list[id](gradOut, gradIn, src, len, dst, buf);
#endif

            // Find error
            err = 0;
            for (i = 0; i < len; i++)
            {
                err += fabs(gradOut[i] - deri[i]);
            }

            printf("=== Test %s derivative ===\n", cnn_activ_name[id]);
            printf("deri:\n");
            print_mat(deri, len, 1);
            printf("\n");
            printf("grad:\n");
            print_mat(gradOut, len, 1);
            printf("\n");
            printf("Sum of error: %lf\n", err);
            printf("\n");
        }
    }

    return 0;
}

void matmul(float* matA, int rowsA, int colsA, float* matB, int rowsB,
            int colsB, float* matC, int rowsC, int colsC)
{
    int i, j, k;
    float tmp;

    // Check argument
    assert((rowsA == rowsC && colsB == colsC && colsA == rowsB) &&
           "Invalid argument for matrix multiplication");

    // Matrix multiplication
    for (i = 0; i < rowsC; i++)
    {
        for (j = 0; j < colsC; j++)
        {
            tmp = 0;
            for (k = 0; k < colsA; k++)
            {
                tmp += matA[i * colsA + k] * matB[k * colsB + j];
            }

            matC[i * colsC + j] = tmp;
        }
    }
}
