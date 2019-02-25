#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cnn.h>

#include "test.h"

#define BATCH 100
#define WIDTH 32
#define HEIGHT 32
#define CHANNEL 3
#define OUTPUTS 10

int main()
{
    int i;
    int inCols;
    float err;

    cnn_t cnn;
    cnn_config_t cfg;

    float* in;
    float* out;

    float* inBat;
    float* outBat;

    test(cnn_init());

    test(cnn_config_create(&cfg));
    test(cnn_config_set_input_size(cfg, WIDTH, HEIGHT, CHANNEL));

    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, CNN_DIM_2D, 1, 3));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0));
    test(cnn_config_append_pooling(cfg, CNN_DIM_2D, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, CNN_DIM_2D, 1, 3));
    test(cnn_config_append_batchnorm(cfg, 1.0, 0.0));
    test(cnn_config_append_pooling(cfg, CNN_DIM_2D, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_texture(cfg, CNN_RELU, 12));
    test(cnn_config_append_full_connect(cfg, 128));
    test(cnn_config_append_dropout(cfg, 0.5));
    test(cnn_config_append_full_connect(cfg, OUTPUTS));
    test(cnn_config_append_activation(cfg, CNN_SOFTMAX));

    test(cnn_create(&cnn, cfg));
    cnn_rand_network(cnn);

    inCols = WIDTH * HEIGHT * CHANNEL;
    in = calloc(sizeof(float), inCols);
    out = calloc(sizeof(float), OUTPUTS);
    inBat = calloc(sizeof(float), BATCH * inCols);
    outBat = calloc(sizeof(float), BATCH * OUTPUTS);
    if (in == NULL || out == NULL || inBat == NULL || outBat == NULL)
    {
        printf("Memory allocation failed\n");
        return -1;
    }

    for (i = 0; i < inCols; i++)
    {
        in[i] = (float)rand() / (float)RAND_MAX;
    }

    cnn_forward(cnn, in, out);

    for (i = 0; i < BATCH; i++)
    {
        memcpy(&inBat[i * inCols], in, sizeof(float) * inCols);
    }

    test(cnn_resize_batch(&cnn, BATCH));
    cnn_forward(cnn, inBat, outBat);

    err = 0;
    for (i = 0; i < BATCH * OUTPUTS; i++)
    {
        err += fabs(outBat[i] - out[i % OUTPUTS]);
    }

    printf("Error sum: %g\n", err);

    return 0;
}
