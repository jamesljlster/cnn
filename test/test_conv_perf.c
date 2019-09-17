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
    int kSize;
    int chIn, chOut;
    int imgWidth, imgHeight;
    int batch;
    int loops;
    int samples;

    struct timespec timeHold;
    float fwCost, bpCost;

    union CNN_LAYER layer[3] = {0};

    cnn_config_t cfg = NULL;

    // Parse argument
    if (argc < 9)
    {
        printf(
            "Usage: %s <kSize> <chIn> <chOut> <imgWidth> <imgHeight> <batch> "
            "<loops> <samples>\n",
            argv[0]);
        return -1;
    }

    i = 1;
    kSize = atoi(argv[i++]);
    chIn = atoi(argv[i++]);
    chOut = atoi(argv[i++]);
    imgWidth = atoi(argv[i++]);
    imgHeight = atoi(argv[i++]);
    batch = atoi(argv[i++]);
    loops = atoi(argv[i++]);
    samples = atoi(argv[i++]);

    printf("kSize, chIn, chOut, imgWidth, imgHeight, batch, loops, samples\n");
    printf("%d, %d, %d, %d, %d, %d, %d, %d\n\n", kSize, chIn, chOut, imgWidth,
           imgHeight, batch, loops, samples);

    // Create cnn
    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_batch_size(cfg, batch));
    test(cnn_config_set_input_size(cfg, imgWidth, imgHeight, chIn));

    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_SAME, CNN_DIM_2D, chOut,
                                       kSize));

    // Allocate cnn layer
    test(cnn_layer_activ_alloc(
        &layer[1].activ, (struct CNN_CONFIG_LAYER_ACTIV*)&cfg->layerCfg[1],
        imgWidth, imgHeight, chIn, cfg->batch));
    test(cnn_layer_conv_alloc(&layer[2].conv,
                              (struct CNN_CONFIG_LAYER_CONV*)&cfg->layerCfg[2],
                              imgWidth, imgHeight, chIn, cfg->batch));

#ifdef CNN_WITH_CUDA
    test(cnn_cudnn_ws_alloc());
#endif

    // Performance test
    printf("Forward (ms), Backward (ms)\n");
    for (int s = 0; s < samples; s++)
    {
        // Forward
        timeHold = hold_time();
        for (int i = 0; i < loops; i++)
        {
            cnn_forward_conv(layer, cfg, 2);
        }
        fwCost = get_time_cost(timeHold);

        // BP
        timeHold = hold_time();
        for (int i = 0; i < loops; i++)
        {
            cnn_backward_conv(layer, cfg, 2);
        }
        bpCost = get_time_cost(timeHold);

        printf("%g, %g\n", fwCost, bpCost);
    }

    return 0;
}
