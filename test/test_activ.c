#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#include "test.h"

#define SIZE 3
#define BATCH 2

int main(int argc, char* argv[])
{
    int id = 0;
    int size;
    union CNN_LAYER layer[3];

    float src[SIZE * BATCH] = {
        -1, 0, 1,  // batch 1
        -1, 1, 2   // batch 2
    };
    float gradIn[SIZE * BATCH] = {
        -1, 0, 1,  // batch 1
        -1, 1, 2   // batch 2
    };

    cnn_config_t cfg = NULL;

    // Parse argument
    if (argc < 2)
    {
        printf("Usage: %s <activ_id>\n", argv[0]);
        return -1;
    }

    id = atoi(argv[1]);
    printf("Using activation id: %d\n\n", id);

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_batch_size(cfg, BATCH));
    test(cnn_config_set_input_size(cfg, SIZE, 1, 1));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_activation(cfg, id));

    // Print information
    printf("src:\n");
    print_img(src, SIZE, 1, 1, cfg->batch);
    printf("\n");

    printf("gradIn:\n");
    print_img(gradIn, SIZE, 1, 1, cfg->batch);
    printf("\n");

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        SIZE, 1, 1, cfg->batch));
    test(cnn_layer_activ_alloc(
        &layer[2].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[2],
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
        cnn_forward_activ(layer, cfg, 2);

        printf("Activation output:\n");
        print_img_net(layer[2].outMat.data.mat, layer[2].outMat.width,
                      layer[2].outMat.height, layer[2].outMat.channel,
                      cfg->batch);
        printf("\n");
    }

    // BP
    for (int i = 0; i < 2; i++)
    {
        printf("***** BP #%d *****\n", i + 1);
        cnn_backward_activ(layer, cfg, 2);

        printf("Activation layer gradient:\n");
        print_img_net(layer[2].outMat.data.grad, layer[2].outMat.width,
                      layer[2].outMat.height, layer[2].outMat.channel,
                      cfg->batch);
        printf("\n");

        printf("Previous layer gradient:\n");
        print_img_net(layer[1].outMat.data.grad, layer[1].outMat.width,
                      layer[1].outMat.height, layer[1].outMat.channel,
                      cfg->batch);
        printf("\n");
    }

    return 0;
}
