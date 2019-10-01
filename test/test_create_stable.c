#include <stdio.h>

#include <cnn.h>
#include <cnn_types.h>

#include "test.h"

#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define BATCH 7

int main()
{
    cnn_config_t cfg = NULL;
    cnn_t cnn = NULL;

    test(cnn_init());
    test(cnn_config_create(&cfg));

    // Set config
    test(cnn_config_set_input_size(cfg, INPUT_WIDTH, INPUT_HEIGHT, 1));
    test(cnn_config_set_batch_size(cfg, BATCH));

    test(cnn_config_append_texture(cfg, CNN_SIGMOID, 9, 2.7183));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 2, 3, 3));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0, 0.001));
    test(cnn_config_append_pooling(cfg, 2, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 2, 6, 3));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0, 0.001));
    test(cnn_config_append_pooling(cfg, 2, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_full_connect(cfg, 16));
    test(cnn_config_append_dropout(cfg, 0.5));
    test(cnn_config_append_full_connect(cfg, 2));
    test(cnn_config_append_activation(cfg, CNN_SOFTMAX));

    while (1)
    {
        // Create cnn
        test(cnn_create(&cnn, cfg));

        // Cleanup
        cnn_delete(cnn);
    }

    cnn_config_delete(cfg);
    cnn_deinit();

    return 0;
}
