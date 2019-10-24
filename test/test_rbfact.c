#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>
#include <cnn_rbfact.h>

#define FLOAT_FMT "%+.7e"
#include "test.h"

#define CH_IN 3
#define CH_OUT 2
#define IMG_WIDTH 2
#define IMG_HEIGHT 2
#define BATCH 2

#define EXP_AVG_FACTOR 0.01

int main()
{
    int size;
    union CNN_LAYER layer[3] = {0};

    float src[IMG_WIDTH * IMG_HEIGHT * CH_IN * BATCH] = {
        /* batch 1 */
        1, 0,  //
        0, 3,  //

        2, 3,  //
        0, 3,  //

        4, 0,  //
        3, 2,  //

        /* batch 2 */
        3, 1,  //
        1, 2,  //

        0, 1,  //
        0, 1,  //

        0, 1,  //
        4, 3,  //
    };

    float center[CH_IN * CH_OUT] = {
        0.7, 0.0,  0.0,  //
        0.0, -0.3, 0.0,  //
    };

    float runVar[CH_OUT] = {
        0.7,  //
        0.9   //
    };

    float gradIn[IMG_WIDTH * IMG_HEIGHT * CH_OUT * BATCH] = {
        /* batch 1 */
        1, 3,  //
        4, 1,  //

        3, 2,  //
        4, 1,  //

        /* batch 2 */
        2, 4,  //
        4, 4,  //

        2, 1,  //
        2, 3,  //
    };

    cnn_config_t cfg = NULL;

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_batch_size(cfg, BATCH));
    test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, CH_IN));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_rbfact(cfg, CH_OUT, EXP_AVG_FACTOR));

    // Print information
    print_img_msg("src:", src, IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch);
    print_img_msg("gradIn", gradIn, IMG_WIDTH, IMG_HEIGHT, CH_OUT, cfg->batch);

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch));
    test(cnn_layer_rbfact_alloc(
        &layer[2].rbfact, (struct CNN_CONFIG_LAYER_RBFACT*)&cfg->layerCfg[2],
        IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch));

    // Copy memory
    size =
        sizeof(float) * layer[1].outMat.data.rows * layer[1].outMat.data.cols;
    memcpy_net(layer[1].outMat.data.mat, src, size);

    size = sizeof(float) * layer[2].rbfact.center.rows *
           layer[2].rbfact.center.cols;
    memcpy_net(layer[2].rbfact.center.mat, center, size);

    size = sizeof(float) * layer[2].rbfact.runVar.rows *
           layer[2].rbfact.runVar.cols;
    memcpy_net(layer[2].rbfact.runVar.mat, runVar, size);

    size =
        sizeof(float) * layer[2].outMat.data.rows * layer[2].outMat.data.cols;
    memcpy_net(layer[2].outMat.data.grad, gradIn, size);

    // Forward
    for (int i = 0; i < 2; i++)
    {
        printf("***** Forward #%d *****\n", i + 1);
        cnn_forward_rbfact(layer, cfg, 2);

        print_img_net_msg("RBFAct output:", layer[2].outMat.data.mat, IMG_WIDTH,
                          IMG_HEIGHT, CH_OUT, cfg->batch);
        print_img_net_msg("Saved variance:", layer[2].rbfact.saveVar.mat, 1, 1,
                          CH_OUT, 1);
        print_img_net_msg("Running variance:", layer[2].rbfact.runVar.mat, 1, 1,
                          CH_OUT, 1);
    }

    // BP
    for (int i = 0; i < 2; i++)
    {
        printf("***** BP #%d *****\n", i + 1);
        cnn_backward_rbfact(layer, cfg, 2);

        print_img_net_msg("Center gradient:", layer[2].rbfact.center.grad,
                          CH_IN, 1, CH_OUT, 1);
        print_img_net_msg("Previous layer gradient:", layer[1].outMat.data.grad,
                          layer[1].outMat.width, layer[1].outMat.height,
                          layer[1].outMat.channel, cfg->batch);
    }

    // Recall
    for (int i = 0; i < 2; i++)
    {
        printf("***** Recall #%d *****\n", i + 1);
        cnn_recall_rbfact(layer, cfg, 2);

        print_img_net_msg("RBFAct output:", layer[2].outMat.data.mat, IMG_WIDTH,
                          IMG_HEIGHT, CH_OUT, cfg->batch);
    }

    return 0;
}
