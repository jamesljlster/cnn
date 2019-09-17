#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#include "test.h"

#define SIZE_IN 4
#define SIZE_OUT 3
#define BATCH 2

int main()
{
    int size;
    union CNN_LAYER layer[3] = {0};

    float src[SIZE_IN * BATCH] = {
        0.1, 0.2, 0.3, 0.4,  // batch 1
        0.5, 0.6, 0.7, 0.8   // batch 2
    };

    float gradIn[SIZE_OUT * BATCH] = {
        0.1, 0.2, 0.3,  // batch 1
        0.4, 0.5, 0.6   // batch 2
    };

    float weight[SIZE_IN * SIZE_OUT] = {
        0.1, 0.2, 0.3,  //
        0.4, 0.5, 0.6,  //
        0.7, 0.8, 0.9,  //
        1.0, 1.1, 1.2   //
    };

    float bias[SIZE_OUT] = {0.1, 0.2, 0.3};

    cnn_config_t cfg = NULL;

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_batch_size(cfg, BATCH));
    test(cnn_config_set_input_size(cfg, SIZE_IN, 1, 1));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_full_connect(cfg, SIZE_OUT));

    // Print information
    print_img_msg("src:", src, SIZE_IN, 1, 1, cfg->batch);
    print_img_msg("weight:", weight, SIZE_OUT, SIZE_IN, 1, 1);
    print_img_msg("bias:", bias, SIZE_OUT, 1, 1, 1);
    print_img_msg("gradIn:", gradIn, SIZE_OUT, 1, 1, cfg->batch);

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        SIZE_IN, 1, 1, cfg->batch));
    test(cnn_layer_fc_alloc(&layer[2].fc,
                            (struct CNN_CONFIG_LAYER_FC*)&cfg->layerCfg[2],
                            SIZE_IN, 1, 1, cfg->batch));

#ifdef CNN_WITH_CUDA
    test(cnn_cudnn_ws_alloc());
#endif

    // Copy memory
    size =
        sizeof(float) * layer[1].outMat.data.rows * layer[1].outMat.data.cols;
    memcpy_net(layer[1].outMat.data.mat, src, size);

    size =
        sizeof(float) * layer[2].outMat.data.rows * layer[2].outMat.data.cols;
    memcpy_net(layer[2].outMat.data.grad, gradIn, size);

    size = sizeof(float) * layer[2].fc.weight.rows * layer[2].fc.weight.cols;
    memcpy_net(layer[2].fc.weight.mat, weight, size);

    size = sizeof(float) * layer[2].fc.bias.rows * layer[2].fc.bias.cols;
    memcpy_net(layer[2].fc.bias.mat, bias, size);

    // Forward
    for (int i = 0; i < 2; i++)
    {
        printf("***** Forward #%d *****\n", i + 1);
        cnn_forward_fc(layer, cfg, 2);

        print_img_net_msg("FC output:", layer[2].outMat.data.mat,
                          layer[2].outMat.width, layer[2].outMat.height,
                          layer[2].outMat.channel, cfg->batch);
    }

    // BP
    for (int i = 0; i < 2; i++)
    {
        printf("***** BP #%d *****\n", i + 1);
        cnn_backward_fc(layer, cfg, 2);

        print_img_net_msg("FC layer gradient:", layer[2].outMat.data.grad,
                          layer[2].outMat.width, layer[2].outMat.height,
                          layer[2].outMat.channel, cfg->batch);
        print_img_net_msg("Weight gradient:", layer[2].fc.weight.grad,
                          layer[2].fc.weight.cols, layer[2].fc.weight.rows, 1,
                          1);
        print_img_net_msg("Bias gradient:", layer[2].fc.bias.grad,
                          layer[2].fc.bias.cols, layer[2].fc.bias.rows, 1, 1);
        print_img_net_msg("Previous layer gradient:", layer[1].outMat.data.grad,
                          layer[1].outMat.width, layer[1].outMat.height,
                          layer[1].outMat.channel, cfg->batch);
    }

    return 0;
}
