#ifndef __TEST_H__
#define __TEST_H__

#include <assert.h>
#include <stdlib.h>
#include <time.h>

#include <cnn_config.h>
#include <cnn_macro.h>

#ifdef CNN_WITH_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#endif

#define test(func)                                                  \
    {                                                               \
        int __retVal = func;                                        \
        if (__retVal < 0)                                           \
        {                                                           \
            fprintf(stderr, "%s(), %d: %s failed with error: %d\n", \
                    __FUNCTION__, __LINE__, #func, __retVal);       \
            assert(0);                                              \
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
        assert(0);                                                          \
    }

#ifdef CNN_WITH_CUDA
#define test_cu(func)                                               \
    {                                                               \
        cudaError_t ret = func;                                     \
        if (ret != cudaSuccess)                                     \
        {                                                           \
            fprintf(stderr, "%s(), %d: %s failed with error: %d\n", \
                    __FUNCTION__, __LINE__, #func, ret);            \
            assert(0);                                              \
        }                                                           \
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
            assert(0);                                                        \
        }                                                                     \
    }
#endif

#ifndef FLOAT_FMT
#define FLOAT_FMT "%g"
#endif

void print_mat(float* src, int rows, int cols)
{
    int i, j;
    for (i = 0; i < rows; i++)
    {
        printf(" | ");
        for (j = 0; j < cols; j++)
        {
            printf("%+f", src[i * cols + j]);
            if (j < cols - 1)
            {
                printf("  ");
            }
        }
        printf(" |\n");
    }
}

void print_img(float* src, int width, int height, int channel, int batch)
{
    int imSize = width * height;
    int batchSize = imSize * channel;

    for (int b = 0; b < batch; b++)
    {
        int bShift = b * batchSize;

        printf("(batch %d)\n", b);
        for (int ch = 0; ch < channel; ch++)
        {
            int chShift = ch * imSize + bShift;

            printf("[\n");
            for (int h = 0; h < height; h++)
            {
                int shift = h * width + chShift;

                printf("[");
                for (int w = 0; w < width; w++)
                {
                    printf(FLOAT_FMT, src[shift + w]);
                    if (w < width - 1)
                    {
                        printf(", ");
                    }
                    else
                    {
                        printf("]\n");
                    }
                }
            }
            printf("]\n");
        }
    }
}

void print_img_int(int* src, int width, int height, int channel, int batch)
{
    int imSize = width * height;
    int batchSize = imSize * channel;

    for (int b = 0; b < batch; b++)
    {
        int bShift = b * batchSize;

        printf("(batch %d)\n", b);
        for (int ch = 0; ch < channel; ch++)
        {
            int chShift = ch * imSize + bShift;

            printf("[\n");
            for (int h = 0; h < height; h++)
            {
                int shift = h * width + chShift;

                printf("[");
                for (int w = 0; w < width; w++)
                {
                    printf("%d", src[shift + w]);
                    if (w < width - 1)
                    {
                        printf(", ");
                    }
                    else
                    {
                        printf("]\n");
                    }
                }
            }
            printf("]\n");
        }
    }
}

#ifdef CNN_WITH_CUDA
void print_img_cu(float* src, int width, int height, int channel, int batch)
{
    int size = width * height * channel * batch;

    float* buf = NULL;
    alloc(buf, size, float);

    test_cu(cudaMemcpy(buf, src, size * sizeof(float), cudaMemcpyDeviceToHost));
    print_img(buf, width, height, channel, batch);

    free(buf);
}

void print_img_int_cu(int* src, int width, int height, int channel, int batch)
{
    int size = width * height * channel * batch;

    int* buf = NULL;
    alloc(buf, size, int);

    test_cu(cudaMemcpy(buf, src, size * sizeof(int), cudaMemcpyDeviceToHost));
    print_img_int(buf, width, height, channel, batch);

    free(buf);
}
#endif

void (*print_img_net)(float*, int, int, int, int) =
#ifdef CNN_WITH_CUDA
    print_img_cu;
#else
    print_img;
#endif

void (*print_img_int_net)(int*, int, int, int, int) =
#ifdef CNN_WITH_CUDA
    print_img_int_cu;
#else
    print_img_int;
#endif

void print_img_msg(const char* msg, float* src, int width, int height,
                   int channel, int batch)
{
    printf("%s\n", msg);
    print_img(src, width, height, channel, batch);
    printf("\n");
}

void print_img_net_msg(const char* msg, float* src, int width, int height,
                       int channel, int batch)
{
    printf("%s\n", msg);
    print_img_net(src, width, height, channel, batch);
    printf("\n");
}

void print_img_int_msg(const char* msg, int* src, int width, int height,
                       int channel, int batch)
{
    printf("%s\n", msg);
    print_img_int(src, width, height, channel, batch);
    printf("\n");
}

void print_img_int_net_msg(const char* msg, int* src, int width, int height,
                           int channel, int batch)
{
    printf("%s\n", msg);
    print_img_int_net(src, width, height, channel, batch);
    printf("\n");
}

#ifdef CNN_WITH_CUDA
#define memcpy_net(dst, src, size)                                            \
    {                                                                         \
        cudaError_t ret = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice); \
        if (ret != cudaSuccess)                                               \
        {                                                                     \
            fprintf(stderr, "%s(), %d: cudaMemcpy failed with error: %d\n",   \
                    __FUNCTION__, __LINE__, ret);                             \
            assert(0);                                                        \
        }                                                                     \
    }
#else
#define memcpy_net(dst, src, size) memcpy(dst, src, size)
#endif

struct timespec hold_time()
{
    struct timespec tmpTime;

    clock_gettime(CLOCK_MONOTONIC, &tmpTime);
    return tmpTime;
}

float get_time_cost(struct timespec timeHold)
{
    struct timespec tmpTime = hold_time();

    tmpTime.tv_sec -= timeHold.tv_sec;
    tmpTime.tv_nsec -= timeHold.tv_nsec;

    return tmpTime.tv_sec * 1000.0 + (double)tmpTime.tv_nsec / 1e+6;
}

#ifdef CNN_WITH_CUDA
void cudnn_log(cudnnSeverity_t sev, void* udata, const cudnnDebug_t* dbg,
               const char* msg)
{
    // Print time step
    printf("[%d, %d] ", dbg->time_sec, dbg->time_usec);

    // Print severity message
    const char* sevMsg = "";
    switch (sev)
    {
        case CUDNN_SEV_FATAL:
            sevMsg = "Fatal";
            break;

        case CUDNN_SEV_ERROR:
            sevMsg = "Error";
            break;

        case CUDNN_SEV_WARNING:
            sevMsg = "Warning";
            break;

        case CUDNN_SEV_INFO:
            sevMsg = "Info";
            break;
    }

    printf("[%s]: ", sevMsg);

    // Print message
    int index = 0;
    char ch = -1;
    char preCh = -1;
    while (1)
    {
        if (ch == '\0' && preCh == '\0')
        {
            break;
        }

        preCh = ch;
        ch = msg[index++];
        if (ch == '\0')
        {
            printf("\n");
        }
        printf("%c", ch);
    }
}

void cudnn_log_enable()
{
    cnn_assert_cudnn(cudnnSetCallback(
        CUDNN_SEV_INFO_EN | CUDNN_SEV_ERROR_EN | CUDNN_SEV_WARNING_EN, NULL,
        cudnn_log));
}

#endif

#endif
