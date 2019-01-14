#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#define KERNEL_SIZE 3
#define CH_IN 3
#define CH_OUT 2
#define IMG_WIDTH 4
#define IMG_HEIGHT 4

#define test(func)                                        \
    ret = func;                                           \
    if (ret < 0)                                          \
    {                                                     \
        printf("%s failed with error: %d\n", #func, ret); \
        return -1;                                        \
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

int main()
{
    // int i;
    int ret;
    union CNN_LAYER layer[3];

    // float src[IMG_WIDTH * IMG_HEIGHT * CH_IN] = {0};
    float src[IMG_WIDTH * IMG_HEIGHT * CH_IN] = {
        1, 0, 3, 0,  //
        0, 3, 0, 4,  //
        4, 0, 3, 0,  //
        3, 1, 0, 4,  //

        2, 3, 2, 2,  //
        0, 3, 3, 0,  //
        0, 2, 2, 4,  //
        2, 1, 4, 0,  //

        4, 0, 0, 4,  //
        3, 2, 0, 2,  //
        3, 1, 4, 3,  //
        4, 1, 4, 1   //
    };

    // float kernel[CH_OUT * CH_IN * KERNEL_SIZE * KERNEL_SIZE] = {0};
    float kernel[CH_OUT * CH_IN * KERNEL_SIZE * KERNEL_SIZE] = {
        0, 3, 4, 0, 1, 2, 2, 4, 1,  //
        1, 3, 0, 0, 2, 0, 4, 4, 2,  //
        0, 4, 4, 1, 3, 4, 2, 4, 4,  //

        3, 1, 0, 4, 4, 3, 0, 1, 1,  //
        3, 2, 3, 0, 4, 2, 1, 2, 0,  //
        2, 0, 2, 1, 0, 3, 1, 4, 4   //
    };

    // float bias[CH_OUT * (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH -
    // KERNEL_SIZE + 1)] = {0};
    float bias[CH_OUT] = {
        1, 2  //
    };

    // float desire[CH_OUT * (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH -
    // KERNEL_SIZE + 1)] = {0};
    float desire[CH_OUT * (IMG_WIDTH - KERNEL_SIZE + 1) *
                 (IMG_WIDTH - KERNEL_SIZE + 1)] = {
        3, 1, 4, 2,  //
        4, 4, 4, 4   //
    };

    cnn_config_t cfg = NULL;

    // Create cnn
    test(cnn_config_create(&cfg));
    test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, CH_IN));
    test(cnn_config_set_layers(cfg, 3));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, CNN_DIM_2D, CH_OUT,
                                       KERNEL_SIZE));

    // Rand
    /*
    srand(time(NULL));
    for(i = 0; i < IMG_WIDTH * IMG_HEIGHT * CH_IN; i++)
    {
        src[i] = rand() % 5;
    }

    for(i = 0; i < CH_OUT * CH_IN * KERNEL_SIZE * KERNEL_SIZE; i++)
    {
        kernel[i] = rand() % 5;
    }

    for(i = 0; i < CH_OUT * (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH -
    KERNEL_SIZE + 1); i++)
    {
        bias[i] = rand() % 5;
        desire[i] = rand() % 5;
    }
    */

    // Print information
    printf("src:\n");
    print_img(src, IMG_WIDTH, IMG_HEIGHT, CH_IN);
    printf("\n");

    printf("kernel:\n");
    print_img(kernel, KERNEL_SIZE, KERNEL_SIZE * CH_IN, CH_OUT);
    printf("\n");

    printf("bias:\n");
    print_img(bias, 1, 1, CH_OUT);
    printf("\n");

    printf("desire:\n");
    print_img(desire,
              (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1), 1,
              CH_OUT);
    printf("\n");

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(&layer[1].activ, IMG_WIDTH, IMG_HEIGHT, CH_IN, 1,
                               CNN_RELU));
    test(cnn_layer_conv_alloc(&layer[2].conv, IMG_WIDTH, IMG_HEIGHT, CH_IN,
                              CH_OUT, KERNEL_SIZE, 1));

    // Copy memory
    memcpy(
        layer[1].outMat.data.mat, src,
        sizeof(float) * layer[1].outMat.data.rows * layer[1].outMat.data.cols);
    memcpy(
        layer[2].conv.kernel.mat, kernel,
        sizeof(float) * layer[2].conv.kernel.rows * layer[2].conv.kernel.cols);

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
    memcpy(layer[2].conv.bias.mat, bias,
           sizeof(float) * layer[2].conv.bias.rows * layer[2].conv.bias.cols);
#endif

    // Forward
    printf("***** Forward *****\n");
    cnn_forward_conv(layer, cfg, 2);

    printf("Convolution output:\n");
    print_img(layer[2].outMat.data.mat, layer[2].outMat.width,
              layer[2].outMat.height, layer[2].outMat.channel);
    printf("\n");

    printf("***** Forward #2 *****\n");
    cnn_forward_conv(layer, cfg, 2);

    printf("Convolution output:\n");
    print_img(layer[2].outMat.data.mat, layer[2].outMat.width,
              layer[2].outMat.height, layer[2].outMat.channel);
    printf("\n");

    // BP
    printf("***** BP *****\n");
    memcpy(
        layer[2].outMat.data.grad, desire,
        sizeof(float) * layer[2].outMat.data.rows * layer[2].outMat.data.cols);
    cnn_backward_conv(layer, cfg, 2);

    printf("Convolution layer gradient:\n");
    print_img(layer[2].outMat.data.grad, layer[2].outMat.width,
              layer[2].outMat.height, layer[2].outMat.channel);
    printf("\n");

    printf("Previous layer gradient:\n");
    print_img(layer[1].outMat.data.grad, layer[1].outMat.width,
              layer[1].outMat.height, layer[1].outMat.channel);
    printf("\n");

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
    printf("Bias gradient:\n");
    print_img(layer[2].conv.bias.grad, 1, 1, CH_OUT);
    printf("\n");
#endif

    printf("Kernel gradient:\n");
    print_img(layer[2].conv.kernel.grad, KERNEL_SIZE, KERNEL_SIZE * CH_IN,
              CH_OUT);
    printf("\n");

    printf("***** BP #2 *****\n");
    memcpy(
        layer[2].outMat.data.grad, desire,
        sizeof(float) * layer[2].outMat.data.rows * layer[2].outMat.data.cols);
    cnn_backward_conv(layer, cfg, 2);

    printf("Convolution layer gradient:\n");
    print_img(layer[2].outMat.data.grad, layer[2].outMat.width,
              layer[2].outMat.height, layer[2].outMat.channel);
    printf("\n");

    printf("Previous layer gradient:\n");
    print_img(layer[1].outMat.data.grad, layer[1].outMat.width,
              layer[1].outMat.height, layer[1].outMat.channel);
    printf("\n");

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
    printf("Bias gradient:\n");
    print_img(layer[2].conv.bias.grad, 1, 1, CH_OUT);
    printf("\n");
#endif

    printf("Kernel gradient:\n");
    print_img(layer[2].conv.kernel.grad, KERNEL_SIZE, KERNEL_SIZE * CH_IN,
              CH_OUT);
    printf("\n");

    return 0;
}
