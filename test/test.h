#ifndef __TEST_H__
#define __TEST_H__

#define test(func)                                                  \
    {                                                               \
        int __retVal = func;                                        \
        if (__retVal < 0)                                           \
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

void print_img(float* src, int width, int height, int channel)
{
    int imSize = width * height;

    for (int ch = 0; ch < channel; ch++)
    {
        int chShift = ch * imSize;

        printf("[\n");
        for (int h = 0; h < height; h++)
        {
            int shift = h * width + chShift;

            printf("[");
            for (int w = 0; w < width; w++)
            {
                printf("%g", src[shift + w]);
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

void print_img_int(int* src, int width, int height, int channel)
{
    int imSize = width * height;

    for (int ch = 0; ch < channel; ch++)
    {
        int chShift = ch * imSize;

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

#endif
