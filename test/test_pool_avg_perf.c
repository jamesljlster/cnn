#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#include "test.h"

int main(int argc, char* argv[])
{
    int i = 1;
    int poolSize;
    int imgWidth, imgHeight, channel;
    int batch;
    int loops;
    int samples;

    struct timespec timeHold;
    float fwCost, bpCost;

    union CNN_LAYER layer[3] = {0};

    cnn_config_t cfg = NULL;

    // Parse argument
    if (argc < 8)
    {
        printf(
            "Usage: %s <poolSize> <imgWidth> <imgHeight> <channel> <batch> "
            "<loops> <samples>\n",
            argv[0]);
        return -1;
    }

    i = 1;
    poolSize = atoi(argv[i++]);
    imgWidth = atoi(argv[i++]);
    imgHeight = atoi(argv[i++]);
    channel = atoi(argv[i++]);
    batch = atoi(argv[i++]);
    loops = atoi(argv[i++]);
    samples = atoi(argv[i++]);

    printf("poolSize, imgWidth, imgHeight, channel, batch, loops, samples\n");
    printf("%d, %d, %d, %d, %d, %d, %d\n\n", poolSize, imgWidth, imgHeight,
           channel, batch, loops, samples);

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_batch_size(cfg, batch));
    test(cnn_config_set_input_size(cfg, imgWidth, imgHeight, channel));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_pooling(cfg, CNN_DIM_2D, CNN_POOL_AVG, poolSize));

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        imgWidth, imgHeight, channel, cfg->batch));
    test(cnn_layer_pool_alloc(&layer[2].pool,
                              (struct CNN_CONFIG_LAYER_POOL*)&cfg->layerCfg[2],
                              imgWidth, imgHeight, channel, cfg->batch));

    // Performance test
    printf("Forward (ms), Backward (ms)\n");
    for (int s = 0; s < samples; s++)
    {
        // Forward
        timeHold = hold_time();
        for (int i = 0; i < loops; i++)
        {
            cnn_forward_pool(layer, cfg, 2);
        }
        fwCost = get_time_cost(timeHold);

        // BP
        timeHold = hold_time();
        for (int i = 0; i < loops; i++)
        {
            cnn_backward_pool(layer, cfg, 2);
        }
        bpCost = get_time_cost(timeHold);

        printf("%g, %g\n", fwCost, bpCost);
    }

    return 0;
}
