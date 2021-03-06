#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#include "test.h"

#define POOL_SIZE 2
#define IMG_WIDTH 5
#define IMG_HEIGHT 5
#define CH_IN 3
#define BATCH 2

int main()
{
    int size;
    union CNN_LAYER layer[3] = {0};

    float src[IMG_WIDTH * IMG_HEIGHT * CH_IN * BATCH] = {
        /* batch 1 */
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

        /* batch 2 */
        1, 4, 1, 3, 1,  //
        3, 4, 2, 2, 1,  //
        0, 1, 4, 3, 4,  //
        3, 0, 1, 3, 4,  //
        0, 3, 3, 3, 4,  //

        3, 4, 1, 4, 4,  //
        3, 4, 0, 1, 0,  //
        2, 3, 0, 1, 0,  //
        1, 0, 1, 0, 3,  //
        4, 2, 2, 1, 4,  //

        0, 1, 2, 4, 3,  //
        3, 3, 4, 4, 4,  //
        1, 1, 3, 0, 0,  //
        2, 0, 3, 4, 0,  //
        1, 1, 2, 4, 0   //
    };

    float gradIn[(IMG_WIDTH / POOL_SIZE) * (IMG_HEIGHT / POOL_SIZE) * CH_IN *
                 BATCH] = {
        /* batch 1 */
        4, 1,  //
        2, 3,  //

        0, 2,  //
        3, 1,  //

        4, 2,  //
        1, 0,  //

        /* batch 2 */
        0, 0,  //
        3, 4,  //

        3, 1,  //
        3, 1,  //

        0, 0,  //
        4, 1   //
    };

    cnn_config_t cfg = NULL;

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_batch_size(cfg, BATCH));
    test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, CH_IN));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_pooling(cfg, CNN_DIM_2D, CNN_POOL_AVG, POOL_SIZE));

    // Print information
    print_img_msg("src:", src, IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch);
    print_img_msg("gradIn:", gradIn, IMG_WIDTH / POOL_SIZE,
                  IMG_HEIGHT / POOL_SIZE, CH_IN, cfg->batch);

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch));
    test(cnn_layer_pool_alloc(&layer[2].pool,
                              (struct CNN_CONFIG_LAYER_POOL*)&cfg->layerCfg[2],
                              IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch));

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
        cnn_forward_pool(layer, cfg, 2);

        print_img_net_msg("Pooling output:", layer[2].outMat.data.mat,
                          layer[2].outMat.width, layer[2].outMat.height,
                          layer[2].outMat.channel, cfg->batch);
    }

    // BP
    for (int i = 0; i < 2; i++)
    {
        printf("***** BP #%d *****\n", i + 1);
        cnn_backward_pool(layer, cfg, 2);

        print_img_net_msg("Pooling layer gradient:", layer[2].outMat.data.grad,
                          layer[2].outMat.width, layer[2].outMat.height,
                          layer[2].outMat.channel, cfg->batch);

        print_img_net_msg("Previous layer gradient:", layer[1].outMat.data.grad,
                          layer[1].outMat.width, layer[1].outMat.height,
                          layer[1].outMat.channel, cfg->batch);
    }

    return 0;
}
