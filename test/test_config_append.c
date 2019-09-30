#include <stdio.h>

#include <cnn.h>
#include <cnn_private.h>
#include <cnn_types.h>

#include "test.h"

#define CFG_PATH "test.xml"

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

        case CNN_LAYER_POOL:
            printf("The layer is pooling\n");
            printf("Dimension: %d\n", cfg->layerCfg[layerIndex].pool.dim);
            printf("Size: %d\n", cfg->layerCfg[layerIndex].pool.size);
            break;

        case CNN_LAYER_DROP:
            printf("The layer is dropout\n");
            printf("Rate: %g\n", cfg->layerCfg[layerIndex].drop.rate);
            break;

        case CNN_LAYER_BN:
            printf("The layer is batchnorm\n");
            printf("rInit: %g\n", cfg->layerCfg[layerIndex].bn.rInit);
            printf("bInit: %g\n", cfg->layerCfg[layerIndex].bn.bInit);
            printf("expAvgFactor: %g\n",
                   cfg->layerCfg[layerIndex].bn.expAvgFactor);
            break;

        case CNN_LAYER_TEXT:
            printf("The layer is texture\n");
            printf("activID: %d\n", cfg->layerCfg[layerIndex].text.activId);
            printf("filter: %d\n", cfg->layerCfg[layerIndex].text.filter);
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

    test(cnn_init());
    test(cnn_config_create(&cfg));

    // Check default setting
    printf("=== Default CNN Config ===\n");
    for (i = 0; i < cfg->layers; i++)
    {
        check_cnn_layer_config(cfg, i);
    }

    // Test set config
    test(cnn_config_set_input_size(cfg, 640, 480, 1));

    test(cnn_config_append_texture(cfg, CNN_SIGMOID, 9, 2.7183));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 1, 1, 3));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0, 0.001));
    test(cnn_config_append_pooling(cfg, 2, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 1, 1, 3));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0, 0.001));
    test(cnn_config_append_pooling(cfg, 2, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_full_connect(cfg, 128));
    test(cnn_config_append_dropout(cfg, 0.5));
    test(cnn_config_append_full_connect(cfg, 2));
    test(cnn_config_append_activation(cfg, CNN_SOFTMAX));

    // Check default setting
    printf("=== Modify CNN Config ===\n");
    for (i = 0; i < cfg->layers; i++)
    {
        check_cnn_layer_config(cfg, i);
    }

    // Export config
    test(cnn_config_export(cfg, CFG_PATH));

    // Cleanup
    cnn_config_delete(cfg);
    return 0;
}
