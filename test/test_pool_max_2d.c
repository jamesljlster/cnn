#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#define IMG_WIDTH 5
#define IMG_HEIGHT 5
#define CH_IN 3

#define POOL_SIZE 2

#define test(func)                                        \
    ret = func;                                           \
    if (ret < 0)                                          \
    {                                                     \
        printf("%s failed with error: %d\n", #func, ret); \
        return -1;                                        \
    }

void print_img(float* src, int width, int height, int channel);
void print_img_int(int* src, int width, int height, int channel);

int main()
{
    int ret;

    // float src[IMG_WIDTH * IMG_HEIGHT * CH_IN] = {0};
    float src[IMG_WIDTH * IMG_HEIGHT * CH_IN] = {
        1, 0, 3, 0, 4,  //
        0, 3, 0, 4, 1,  //
        4, 0, 3, 0, 3,  //
        3, 1, 0, 4, 2,  //
        4, 1, 2, 2, 3,  //

        2, 3, 2, 2, 1,  //
        0, 3, 3, 0, 3,  //
        0, 2, 2, 4, 4,  //
        2, 1, 4, 0, 2,  //
        4, 3, 1, 2, 0,  //

        4, 0, 0, 4, 2,  //
        3, 2, 0, 2, 4,  //
        3, 1, 4, 3, 1,  //
        4, 1, 4, 1, 0,  //
        3, 0, 1, 4, 2,  //
    };

    float desire[(IMG_WIDTH / POOL_SIZE) * (IMG_HEIGHT / POOL_SIZE) * CH_IN] = {
        4, 1,  //
        2, 3,  //

        0, 2,  //
        3, 1,  //

        4, 2,  //
        1, 0   //
    };

    cnn_config_t cfg = NULL;
    cnn_t cnn = NULL;

    // Create cnn
    test(cnn_config_create(&cfg));
    test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, CH_IN));
    test(cnn_config_set_layers(cfg, 3));
    test(cnn_config_set_activation(cfg, 1, CNN_RELU));
    test(cnn_config_set_pooling(cfg, 2, CNN_DIM_2D, CNN_POOL_MAX, POOL_SIZE));

    test(cnn_create(&cnn, cfg));

    // Print information
    printf("src:\n");
    print_img(src, IMG_WIDTH, IMG_HEIGHT, CH_IN);
    printf("\n");
    printf("desire:\n");
    print_img(desire, IMG_WIDTH / POOL_SIZE, IMG_HEIGHT / POOL_SIZE, CH_IN);
    printf("\n");

    // Copy memory
    memcpy(cnn->layerList[1].outMat.data.mat, src,
           sizeof(float) * cnn->layerList[1].outMat.data.rows *
               cnn->layerList[1].outMat.data.cols);

    // Forward
    printf("***** Forward *****\n");
    cnn_forward_pool(cnn->layerList, cfg, 2);

    printf("Pooling output:\n");
    print_img(cnn->layerList[2].outMat.data.mat, cnn->layerList[2].outMat.width,
              cnn->layerList[2].outMat.height,
              cnn->layerList[2].outMat.channel);
    printf("\n");

    printf("***** Forward #2 *****\n");
    cnn_forward_pool(cnn->layerList, cfg, 2);

    printf("Pooling output:\n");
    print_img(cnn->layerList[2].outMat.data.mat, cnn->layerList[2].outMat.width,
              cnn->layerList[2].outMat.height,
              cnn->layerList[2].outMat.channel);
    printf("\n");

    // BP
    printf("***** BP *****:\n");
    memcpy(cnn->layerList[2].outMat.data.grad, desire,
           sizeof(float) * cnn->layerList[2].outMat.data.rows *
               cnn->layerList[2].outMat.data.cols);
    cnn_backward_pool(cnn->layerList, cfg, 2);

    printf("Gradient index mat:\n");
    print_img_int(
        cnn->layerList[2].pool.indexMat, cnn->layerList[2].outMat.width,
        cnn->layerList[2].outMat.height, cnn->layerList[2].outMat.channel);
    printf("\n");

    printf("Previous layer gradient:\n");
    print_img(cnn->layerList[1].outMat.data.grad,
              cnn->layerList[1].outMat.width, cnn->layerList[1].outMat.height,
              cnn->layerList[1].outMat.channel);
    printf("\n");

    return 0;
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
