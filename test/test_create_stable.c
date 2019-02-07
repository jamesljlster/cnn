#include <stdio.h>

#include <cnn.h>
#include <cnn_types.h>

#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define BATCH 7

#define test(func)                                        \
    ret = func;                                           \
    if (ret < 0)                                          \
    {                                                     \
        printf("%s failed with error: %d\n", #func, ret); \
        return -1;                                        \
    }

int main()
{
    int ret;
    cnn_config_t cfg = NULL;
    cnn_t cnn = NULL;

    ret = cnn_init();
    if (ret < 0)
    {
        printf("cnn_init() failed with error: %d\n", ret);
        return -1;
    }

    ret = cnn_config_create(&cfg);
    if (ret < 0)
    {
        printf("cnn_config_create() failed with error: %d\n", ret);
        return -1;
    }

    // Set config
    test(cnn_config_set_input_size(cfg, INPUT_WIDTH, INPUT_HEIGHT, 1));
    test(cnn_config_set_batch_size(cfg, BATCH));

    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 2, 3, 3));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0));
    test(cnn_config_append_pooling(cfg, 2, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 2, 6, 3));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0));
    test(cnn_config_append_pooling(cfg, 2, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_full_connect(cfg, 16));
    test(cnn_config_append_dropout(cfg, 0.5));
    test(cnn_config_append_full_connect(cfg, 2));
    test(cnn_config_append_activation(cfg, CNN_SOFTMAX));

    while (1)
    {
        // Create cnn
        ret = cnn_create(&cnn, cfg);
        if (ret < 0)
        {
            printf("cnn_create() failed with error: %d\n", ret);
            return -1;
        }

        // Cleanup
        cnn_delete(cnn);
    }

    cnn_config_delete(cfg);
    cnn_deinit();

    return 0;
}
