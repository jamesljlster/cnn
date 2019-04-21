#include <stdio.h>

#include <cnn.h>
#include <cnn_private.h>
#include <cnn_types.h>

#include "test.h"

void check_cnn_layer_config(cnn_config_t cfg, int layerIndex)
{
    printf("Layer %d config:\n", layerIndex);
    switch (cfg->layerCfg[layerIndex].type)
    {
        case CNN_LAYER_INPUT:
            printf("The layer is input\n");
            break;

        case CNN_LAYER_FC:
            printf("The layer is fully connected\n");
            printf("Size: %d\n", cfg->layerCfg[layerIndex].fc.size);
            break;

        case CNN_LAYER_ACTIV:
            printf("The layer is activation function\n");
            printf("ID: %d\n", cfg->layerCfg[layerIndex].activ.id);
            break;

        case CNN_LAYER_CONV:
            printf("The layer is convolution\n");
            printf("Dimension: %d\n", cfg->layerCfg[layerIndex].conv.dim);
            printf("Size: %d\n", cfg->layerCfg[layerIndex].conv.size);
            break;

        default:
            printf("Not a cnn layer config?!\n");
    }
    printf("\n");
}

int main()
{
    int i;
    cnn_config_t cfg = NULL;
    cnn_config_t cpy = NULL;

    test(cnn_init());
    test(cnn_config_create(&cfg));

    // Test set config
    cnn_config_set_input_size(cfg, 640, 480, 1);

    cnn_config_append_convolution(cfg, CNN_PAD_VALID, 1, 1, 3);
    cnn_config_append_activation(cfg, CNN_RELU);
    cnn_config_append_convolution(cfg, CNN_PAD_VALID, 1, 1, 3);
    cnn_config_append_activation(cfg, CNN_RELU);
    cnn_config_append_full_connect(cfg, 128);
    cnn_config_append_full_connect(cfg, 2);
    cnn_config_append_activation(cfg, CNN_SOFTMAX);

    // Clone cnn setting
    test(cnn_config_clone(&cpy, cfg));

    // Check setting
    printf("=== Src CNN Config ===\n");
    for (i = 0; i < cfg->layers; i++)
    {
        check_cnn_layer_config(cfg, i);
    }

    printf("=== Cpy CNN Config ===\n");
    for (i = 0; i < cfg->layers; i++)
    {
        check_cnn_layer_config(cpy, i);
    }

    // Compare
    test(cnn_config_compare(cfg, cpy));
    printf("cnn_config_compare() passed\n");

    // Cleanup
    cnn_config_delete(cfg);
    cnn_config_delete(cpy);

    return 0;
}
