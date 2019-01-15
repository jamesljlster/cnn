#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#define CH_IN 2
#define CH_OUT 2
#define IMG_WIDTH 4
#define IMG_HEIGHT 1

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
    int ret;
    union CNN_LAYER layer[3];

    float src[IMG_WIDTH * IMG_HEIGHT * CH_IN] = {
        0.0, 0.1, 0.2, 0.3,  //
        0.2, 0.4, 0.6, 0.8   //
    };

    float desire[IMG_WIDTH * IMG_HEIGHT * CH_IN] = {
        0.3, 0.1, 0.4, 0.2,  //
        0.4, 0.2, 0.1, 0.3   //
    };

    cnn_config_t cfg = NULL;

    // Create cnn
    test(cnn_config_create(&cfg));
    test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, CH_IN));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0));

    // Print information
    printf("src:\n");
    print_img(src, IMG_WIDTH, IMG_HEIGHT, CH_IN);
    printf("\n");

    printf("desire:\n");
    print_img(desire, IMG_WIDTH, IMG_HEIGHT, CH_IN);
    printf("\n");

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(&layer[1].activ, IMG_WIDTH, IMG_HEIGHT, CH_IN, 1,
                               CNN_RELU));
    test(cnn_layer_bn_alloc(&layer[2].bn, IMG_WIDTH, IMG_HEIGHT, CH_IN, 1, 1.0,
                            0.0));

    // Copy memory
    memcpy(
        layer[1].outMat.data.mat, src,
        sizeof(float) * layer[1].outMat.data.rows * layer[1].outMat.data.cols);

    // Forward
    printf("***** Forward *****\n");
    cnn_forward_bn(layer, cfg, 2);

    printf("BatchNorm output:\n");
    print_img(layer[2].outMat.data.mat, layer[2].outMat.width,
              layer[2].outMat.height, layer[2].outMat.channel);
    printf("\n");

    printf("***** Forward #2 *****\n");
    cnn_forward_bn(layer, cfg, 2);

    printf("Convolution output:\n");
    print_img(layer[2].outMat.data.mat, layer[2].outMat.width,
              layer[2].outMat.height, layer[2].outMat.channel);
    printf("\n");

    // BP
    printf("***** BP *****\n");
    memcpy(
        layer[2].outMat.data.grad, desire,
        sizeof(float) * layer[2].outMat.data.rows * layer[2].outMat.data.cols);
    cnn_backward_bn(layer, cfg, 2);

    printf("BatchNorm layer gradient:\n");
    print_img(layer[2].outMat.data.grad, layer[2].outMat.width,
              layer[2].outMat.height, layer[2].outMat.channel);
    printf("\n");

    printf("Previous layer gradient:\n");
    print_img(layer[1].outMat.data.grad, layer[1].outMat.width,
              layer[1].outMat.height, layer[1].outMat.channel);
    printf("\n");

    printf("BatchNorm variable gradient:\n");
    print_img(layer[2].bn.bnVar.grad, 2, CH_IN, 1);
    printf("\n");

    printf("***** BP #2 *****\n");
    memcpy(
        layer[2].outMat.data.grad, desire,
        sizeof(float) * layer[2].outMat.data.rows * layer[2].outMat.data.cols);
    cnn_backward_bn(layer, cfg, 2);

    printf("BatchNorm layer gradient:\n");
    print_img(layer[2].outMat.data.grad, layer[2].outMat.width,
              layer[2].outMat.height, layer[2].outMat.channel);
    printf("\n");

    printf("Previous layer gradient:\n");
    print_img(layer[1].outMat.data.grad, layer[1].outMat.width,
              layer[1].outMat.height, layer[1].outMat.channel);
    printf("\n");

    printf("BatchNorm variable gradient:\n");
    print_img(layer[2].bn.bnVar.grad, 2, CH_IN, 1);
    printf("\n");

    return 0;
}
