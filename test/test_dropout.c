#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#include "test.h"

#define SIZE 5
#define BATCH 2

int main()
{
    int size;
    union CNN_LAYER layer[3];

    float src[SIZE * BATCH] = {
        0.1, 0.2, 0.3, 0.4, 0.5,  // batch 1
        0.6, 0.7, 0.8, 0.9, 1.0   // batch 2
    };
    float gradIn[SIZE * BATCH] = {
        0.1, 0.2, 0.3, 0.4, 0.5,  // batch 1
        0.6, 0.7, 0.8, 0.9, 1.0   // batch 2
    };

    cnn_config_t cfg = NULL;

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_batch_size(cfg, BATCH));
    test(cnn_config_set_input_size(cfg, SIZE, 1, 1));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_dropout(cfg, 0.5));

    // Print information
    print_img_msg("src:", src, SIZE, 1, 1, cfg->batch);
    print_img_msg("gradIn:", gradIn, SIZE, 1, 1, cfg->batch);

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        SIZE, 1, 1, cfg->batch));
    test(cnn_layer_drop_alloc(&layer[2].drop,
                              (struct CNN_CONFIG_LAYER_DROP*)&cfg->layerCfg[2],
                              SIZE, 1, 1, cfg->batch));

    // Copy memory
    size =
        sizeof(float) * layer[1].outMat.data.rows * layer[1].outMat.data.cols;
    memcpy_net(layer[1].outMat.data.mat, src, size);

    size =
        sizeof(float) * layer[2].outMat.data.rows * layer[2].outMat.data.cols;
    memcpy_net(layer[2].outMat.data.grad, gradIn, size);

    // Forward
    for (int i = 0; i < 2; i++)
    {
        printf("***** Forward #%d *****\n", i + 1);
        cnn_forward_drop(layer, cfg, 2);

        print_img_net_msg("Dropout output:", layer[2].outMat.data.mat,
                          layer[2].outMat.width, layer[2].outMat.height,
                          layer[2].outMat.channel, cfg->batch);
    }

    // BP
    for (int i = 0; i < 2; i++)
    {
        printf("***** BP #%d *****\n", i + 1);
        cnn_backward_drop(layer, cfg, 2);

        print_img_net_msg("Dropout layer gradient:", layer[2].outMat.data.grad,
                          layer[2].outMat.width, layer[2].outMat.height,
                          layer[2].outMat.channel, cfg->batch);

        print_img_net_msg("Previous layer gradient:", layer[1].outMat.data.grad,
                          layer[1].outMat.width, layer[1].outMat.height,
                          layer[1].outMat.channel, cfg->batch);
    }

    return 0;
}
