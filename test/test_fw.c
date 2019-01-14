#include <stdio.h>
#include <string.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#define KERNEL_SIZE 3
#define IMG_WIDTH 5
#define IMG_HEIGHT 5

#define DST_WIDTH 3
#define DST_HEIGHT 3

#define OUTPUTS 3

#define test(func)                                        \
    ret = func;                                           \
    if (ret < 0)                                          \
    {                                                     \
        printf("%s failed with error: %d\n", #func, ret); \
        return -1;                                        \
    }

void print_img(float* src, int rows, int cols)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            printf("%+5.2f", src[i * cols + j]);
            if (j < cols - 1)
            {
                printf("  ");
            }
            else
            {
                printf("\n");
            }
        }
    }
}

int main()
{
    int i;
    int ret;
    int wRows, wCols;

    cnn_config_t cfg = NULL;
    cnn_t cnn = NULL;

    float kernel[KERNEL_SIZE * KERNEL_SIZE] = {
        -4, -3, -2,  //
        -1, 0,  1,   //
        2,  3,  4    //
    };

    float weight[(IMG_WIDTH - KERNEL_SIZE + 1) *
                 (IMG_HEIGHT - KERNEL_SIZE + 1) * OUTPUTS];
    float bias[OUTPUTS];

    float src[IMG_WIDTH * IMG_HEIGHT];
    float output[OUTPUTS];

    // Set input image
    for (i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
    {
        src[i] = i;
    }

    // Set weight
    wRows = (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_HEIGHT - KERNEL_SIZE + 1);
    wCols = OUTPUTS;
    for (i = 0; i < wRows * wCols; i++)
    {
        weight[i] = (float)i / 10.0;
    }

    // Set bias
    for (i = 0; i < OUTPUTS; i++)
    {
        bias[i] = (float)i / 10.0 + 0.5;
    }

    // Set config
    test(cnn_config_create(&cfg));
    test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, 1));

    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 2, 1, 3));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_pooling(cfg, 2, CNN_POOL_MAX, 2));
    test(cnn_config_append_full_connect(cfg, OUTPUTS));
    test(cnn_config_append_activation(cfg, CNN_SOFTMAX));

    // Create cnn
    test(cnn_create(&cnn, cfg));

    // Set kernel
    memcpy(cnn->layerList[1].conv.kernel.mat, kernel,
           KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    // Set weight
    memcpy(cnn->layerList[4].fc.weight.mat, weight,
           wRows * wCols * sizeof(float));

    // Set bias
    memcpy(cnn->layerList[4].fc.bias.mat, bias, OUTPUTS * sizeof(float));

    // CNN forward
    cnn_forward(cnn, src, output);

    printf("=== Source image ===\n");
    print_img(src, IMG_HEIGHT, IMG_WIDTH);
    printf("\n");

    printf("=== Kernel ===\n");
    print_img(kernel, KERNEL_SIZE, KERNEL_SIZE);
    printf("\n");

    printf("=== Weight ===\n");
    print_img(weight, wRows, wCols);
    printf("\n");

    printf("=== Bias ===\n");
    print_img(bias, 1, OUTPUTS);
    printf("\n");

    printf("=== Output ===\n");
    print_img(output, 1, OUTPUTS);
    printf("\n");

    // Print detail
    printf("*** Network Detail ***\n");
    for (i = 0; i < cnn->cfg.layers; i++)
    {
        printf("=== Layer %d output ===\n", i);
        print_img(cnn->layerList[i].outMat.data.mat,
                  cnn->layerList[i].outMat.data.rows,
                  cnn->layerList[i].outMat.data.cols);
        printf("\n");
    }

    return 0;
}
