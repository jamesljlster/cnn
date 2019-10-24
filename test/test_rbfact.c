#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>
#include <cnn_rbfact.h>

#define FLOAT_FMT "%+.7e"
#include "test.h"

#define W_SIZE 8
#define CH_IN 3
#define CH_OUT 2
#define IMG_WIDTH 2
#define IMG_HEIGHT 2
#define BATCH 2

#define ALPHA 1.0

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

    float centerGrad[CH_IN * CH_OUT];
    float centerBuf[CH_IN * CH_OUT];

    float runVar[CH_OUT] = {
        0.7,  //
        0.9   //
    };

    float saveVar[CH_OUT];
    float varBuf[CH_OUT];

    float output[IMG_WIDTH * IMG_HEIGHT * CH_OUT * BATCH] = {0};

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

    // Print information
    print_img_msg("src:", src, IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch);
    print_img_msg("gradIn", gradIn, IMG_WIDTH, IMG_HEIGHT, CH_OUT, cfg->batch);

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        IMG_WIDTH, IMG_HEIGHT, CH_IN, cfg->batch));

    // Copy memory
    size =
        sizeof(float) * layer[1].outMat.data.rows * layer[1].outMat.data.cols;
    memcpy_net(layer[1].outMat.data.mat, src, size);

    // Forward
    for (int i = 0; i < 2; i++)
    {
        printf("***** Forward #%d *****\n", i + 1);
        cnn_rbfact_forward_training_cpu(output, CH_OUT, src, CH_IN, center,
                                        runVar, saveVar, varBuf, BATCH,
                                        IMG_WIDTH, IMG_HEIGHT, 0.01);

        print_img_net_msg("RBFAct output:", output, IMG_WIDTH, IMG_HEIGHT,
                          CH_OUT, cfg->batch);
        print_img_net_msg("Saved variance:", saveVar, 1, 1, CH_OUT, 1);
        print_img_net_msg("Running variance:", runVar, 1, 1, CH_OUT, 1);
    }

    // BP
    for (int i = 0; i < 2; i++)
    {
        printf("***** BP #%d *****\n", i + 1);
        cnn_rbfact_backward_layer_cpu(layer[1].outMat.data.grad, CH_IN, gradIn,
                                      CH_OUT, src, output, center, saveVar,
                                      BATCH, IMG_HEIGHT, IMG_WIDTH);

        print_img_net_msg("Previous layer gradient:", layer[1].outMat.data.grad,
                          layer[1].outMat.width, layer[1].outMat.height,
                          layer[1].outMat.channel, cfg->batch);
    }

    // Recall
    for (int i = 0; i < 2; i++)
    {
        printf("***** Recall #%d *****\n", i + 1);
        cnn_rbfact_forward_inference_cpu(output, CH_OUT, src, CH_IN, center,
                                         runVar, BATCH, IMG_HEIGHT, IMG_WIDTH);

        print_img_net_msg("RBFAct output:", output, IMG_WIDTH, IMG_HEIGHT,
                          CH_OUT, cfg->batch);
    }

    return 0;
}
