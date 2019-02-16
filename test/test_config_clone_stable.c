#include <stdio.h>

#include <cnn.h>
#include <cnn_types.h>

#include "test.h"

int main()
{
    cnn_config_t cfg = NULL;
    cnn_config_t cpy = NULL;

    test(cnn_init());
    test(cnn_config_create(&cfg));

    // Test set config
    test(cnn_config_set_input_size(cfg, 640, 480, 1));

    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 1, 1, 3));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 1, 1, 3));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_full_connect(cfg, 128));
    test(cnn_config_append_full_connect(cfg, 2));
    test(cnn_config_append_activation(cfg, CNN_SOFTMAX));

    // Clone cnn setting
    while (1)
    {
        test(cnn_config_clone(&cpy, cfg));
        cnn_config_delete(cpy);
    }

    // Cleanup
    cnn_config_delete(cfg);

    return 0;
}
