#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#include "test.h"

int main(int argc, char* argv[])
{
    int i;
    int channel;
    int imgWidth, imgHeight;
    int batch;
    int loops;
    int samples;

    struct timespec timeHold;
    float fwCost, bpCost;

    union CNN_LAYER layer[3];

    cnn_config_t cfg = NULL;

    // Parse argument
    if (argc < 7)
    {
        printf(
            "Usage: %s <channel> <imgWidth> <imgHeight> <batch> <loops> "
            "<samples>\n",
            argv[0]);
        return -1;
    }

    i = 1;
    channel = atoi(argv[i++]);
    imgWidth = atoi(argv[i++]);
    imgHeight = atoi(argv[i++]);
    batch = atoi(argv[i++]);
    loops = atoi(argv[i++]);
    samples = atoi(argv[i++]);

    printf("channel, imgWidth, imgHeight, batch, loops, samples\n");
    printf("%d, %d, %d, %d, %d, %d\n\n", channel, imgWidth, imgHeight, batch,
           loops, samples);

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_batch_size(cfg, batch));
    test(cnn_config_set_input_size(cfg, imgWidth, imgHeight, channel));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_batchnorm(cfg, 0.87, 0.03));

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        imgWidth, imgHeight, channel, cfg->batch));
    test(cnn_layer_bn_alloc(&layer[2].bn,
                            (struct CNN_CONFIG_LAYER_BN*)&cfg->layerCfg[2],
                            imgWidth, imgHeight, channel, cfg->batch));

    // Performance test
    printf("Forward (ms), Backward (ms)\n");
    for (int s = 0; s < samples; s++)
    {
        // Forward
        timeHold = hold_time();
        for (int i = 0; i < loops; i++)
        {
            cnn_forward_bn(layer, cfg, 2);
        }
        fwCost = get_time_cost(timeHold);

        // BP
        timeHold = hold_time();
        for (int i = 0; i < loops; i++)
        {
            cnn_backward_bn(layer, cfg, 2);
        }
        bpCost = get_time_cost(timeHold);

        printf("%g, %g\n", fwCost, bpCost);
    }

    return 0;
}
