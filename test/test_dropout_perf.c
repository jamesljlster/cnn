#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#include "test.h"

int main(int argc, char* argv[])
{
    int i;
    int size;
    int batch;
    int loops;
    int samples;

    struct timespec timeHold;
    float fwCost, bpCost;

    union CNN_LAYER layer[3];

    cnn_config_t cfg = NULL;

    // Parse argument
    if (argc < 5)
    {
        printf("Usage: %s <size> <batch> <loops> <samples>\n", argv[0]);
        return -1;
    }

    i = 1;
    size = atoi(argv[i++]);
    batch = atoi(argv[i++]);
    loops = atoi(argv[i++]);
    samples = atoi(argv[i++]);

    printf("size, batch, loops, samples\n");
    printf("%d, %d, %d, %d\n\n", size, batch, loops, samples);

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_batch_size(cfg, batch));
    test(cnn_config_set_input_size(cfg, size, 1, 1));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_dropout(cfg, 0.5));

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        size, 1, 1, cfg->batch));
    test(cnn_layer_drop_alloc(&layer[2].drop,
                              (struct CNN_CONFIG_LAYER_DROP*)&cfg->layerCfg[2],
                              size, 1, 1, cfg->batch));

    // Performance test
    printf("Forward (ms), Backward (ms)\n");
    for (int s = 0; s < samples; s++)
    {
        // Forward
        timeHold = hold_time();
        for (int i = 0; i < loops; i++)
        {
            cnn_forward_drop(layer, cfg, 2);
        }
        fwCost = get_time_cost(timeHold);

        // BP
        timeHold = hold_time();
        for (int i = 0; i < 2; i++)
        {
            cnn_backward_drop(layer, cfg, 2);
        }
        bpCost = get_time_cost(timeHold);

        printf("%g, %g\n", fwCost, bpCost);
    }

    return 0;
}
