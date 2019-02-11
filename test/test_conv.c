#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#include "test.h"

#define KERNEL_SIZE 3
#define CH_IN 3
#define CH_OUT 2
#define IMG_WIDTH 4
#define IMG_HEIGHT 4
#define BATCH 2

int main()
{
    int size;
    union CNN_LAYER layer[3];

    float src[IMG_WIDTH * IMG_HEIGHT * CH_IN * BATCH] = {
        /* batch 1 */
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
        4, 1, 4, 1,  //

        /* batch 2 */
        3, 1, 3, 1,  //
        1, 2, 4, 2,  //
        4, 2, 2, 4,  //
        2, 1, 3, 3,  //

        0, 1, 4, 4,  //
        0, 1, 3, 3,  //
        1, 2, 1, 1,  //
        0, 4, 2, 0,  //

        0, 1, 2, 1,  //
        4, 3, 4, 4,  //
        1, 1, 3, 0,  //
        3, 4, 4, 2   //
    };

    float kernel[CH_OUT * CH_IN * KERNEL_SIZE * KERNEL_SIZE] = {
        0, 3, 4, 0, 1, 2, 2, 4, 1,  //
        1, 3, 0, 0, 2, 0, 4, 4, 2,  //
        0, 4, 4, 1, 3, 4, 2, 4, 4,  //

        3, 1, 0, 4, 4, 3, 0, 1, 1,  //
        3, 2, 3, 0, 4, 2, 1, 2, 0,  //
        2, 0, 2, 1, 0, 3, 1, 4, 4   //
    };

    float bias[CH_OUT] = {
        1, 2  //
    };

    float gradIn[IMG_WIDTH * IMG_HEIGHT * CH_OUT * BATCH] = {
        /* batch 1 */
        0, 0, 0, 0,  //
        0, 3, 1, 0,  //
        0, 4, 2, 0,  //
        0, 0, 0, 0,  //

        0, 0, 0, 0,  //
        0, 4, 4, 0,  //
        0, 4, 4, 0,  //
        0, 0, 0, 0,  //

        /* batch 2 */
        2, 4, 1, 1,  //
        4, 4, 3, 1,  //
        4, 2, 3, 4,  //
        0, 1, 4, 3,  //

        2, 1, 4, 0,  //
        2, 3, 2, 0,  //
        3, 1, 1, 1,  //
        0, 2, 2, 3   //
    };

    cnn_config_t cfg = NULL;

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_batch_size(cfg, BATCH));
    test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, CH_IN));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_SAME, CNN_DIM_2D, CH_OUT,
                                       KERNEL_SIZE));

    // Print information
    print_img_msg("src:", src, IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch);
    print_img_msg("kernel:", kernel, KERNEL_SIZE, KERNEL_SIZE * CH_IN, CH_OUT,
                  1);
    print_img_msg("bias:", bias, 1, 1, CH_OUT, 1);
    print_img_msg("gradIn", gradIn, IMG_WIDTH, IMG_HEIGHT, CH_OUT, cfg->batch);

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch));
    test(cnn_layer_conv_alloc(&layer[2].conv,
                              (struct CNN_CONFIG_LAYER_CONV*)&cfg->layerCfg[2],
                              IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch));

    // Copy memory
    size =
        sizeof(float) * layer[1].outMat.data.rows * layer[1].outMat.data.cols;
    memcpy_net(layer[1].outMat.data.mat, src, size);

    size =
        sizeof(float) * layer[2].conv.kernel.rows * layer[2].conv.kernel.cols;
    memcpy_net(layer[2].conv.kernel.mat, kernel, size);

    size = sizeof(float) * layer[2].conv.bias.rows * layer[2].conv.bias.cols;
    memcpy_net(layer[2].conv.bias.mat, bias, size);

    size =
        sizeof(float) * layer[2].outMat.data.rows * layer[2].outMat.data.cols;
    memcpy_net(layer[2].outMat.data.grad, gradIn, size);

    // Forward
    for (int i = 0; i < 2; i++)
    {
        printf("***** Forward #%d *****\n", i + 1);
        cnn_forward_conv(layer, cfg, 2);

        print_img_net_msg("Convolution output:", layer[2].outMat.data.mat,
                          layer[2].outMat.width, layer[2].outMat.height,
                          layer[2].outMat.channel, cfg->batch);
    }

    // BP
    for (int i = 0; i < 2; i++)
    {
        printf("***** BP #%d *****\n", i + 1);
        cnn_backward_conv(layer, cfg, 2);

        print_img_net_msg(
            "Convolution layer gradient:", layer[2].outMat.data.grad,
            layer[2].outMat.width, layer[2].outMat.height,
            layer[2].outMat.channel, cfg->batch);
        print_img_net_msg("Previous layer gradient:", layer[1].outMat.data.grad,
                          layer[1].outMat.width, layer[1].outMat.height,
                          layer[1].outMat.channel, cfg->batch);
        print_img_net_msg("Bias gradient:", layer[2].conv.bias.grad, 1, 1,
                          CH_OUT, 1);
        print_img_net_msg("Kernel gradient:", layer[2].conv.kernel.grad,
                          KERNEL_SIZE, KERNEL_SIZE * CH_IN, CH_OUT, 1);
    }

    return 0;
}
