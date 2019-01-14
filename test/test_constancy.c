#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cnn.h>

#define BATCH 100
#define WIDTH 32
#define HEIGHT 32
#define CHANNEL 3
#define OUTPUTS 10

#define test(func)                                        \
    ret = func;                                           \
    if (ret < 0)                                          \
    {                                                     \
        printf("%s failed with error: %d\n", #func, ret); \
        return -1;                                        \
    }

int main()
{
    int i;
    int inCols;
    int ret;
    float err;

    cnn_t cnn;
    cnn_config_t cfg;

    float* inBat;
    float* outBat;
    float* outBatTest;

    test(cnn_config_create(&cfg));
    test(cnn_config_set_input_size(cfg, WIDTH, HEIGHT, CHANNEL));
    test(cnn_config_set_batch_size(cfg, BATCH));

    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, CNN_DIM_2D, 3, 3));
    test(cnn_config_append_pooling(cfg, CNN_DIM_2D, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, CNN_DIM_2D, 6, 3));
    test(cnn_config_append_pooling(cfg, CNN_DIM_2D, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_full_connect(cfg, 256));
    test(cnn_config_append_dropout(cfg, 0.5));
    test(cnn_config_append_full_connect(cfg, 128));
    test(cnn_config_append_dropout(cfg, 0.5));
    test(cnn_config_append_full_connect(cfg, OUTPUTS));
    test(cnn_config_append_activation(cfg, CNN_SOFTMAX));

    test(cnn_create(&cnn, cfg));
    cnn_rand_network(cnn);

    inCols = WIDTH * HEIGHT * CHANNEL;
    inBat = calloc(sizeof(float), BATCH * inCols);
    outBat = calloc(sizeof(float), BATCH * OUTPUTS);
    outBatTest = calloc(sizeof(float), BATCH * OUTPUTS);
    if (inBat == NULL || outBat == NULL || outBatTest == NULL)
    {
        printf("Memory allocation failed\n");
        return -1;
    }

    for (i = 0; i < inCols * BATCH; i++)
    {
        inBat[i] = (float)rand() / (float)RAND_MAX;
    }

    cnn_forward(cnn, inBat, outBat);
    cnn_forward(cnn, inBat, outBatTest);

    err = 0;
    for (i = 0; i < BATCH * OUTPUTS; i++)
    {
        err += fabs(outBat[i] - outBatTest[i]);
    }

    printf("Error sum: %g\n", err);

    return 0;
}
