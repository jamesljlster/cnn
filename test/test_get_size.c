#include <cnn.h>
#include <stdio.h>

#include "test.h"

#define WIDTH 75
#define HEIGHT 125
#define CHANNEL 3

int size_cmp(int w, int h, int ch, int cmpW, int cmpH, int cmpCh)
{
    int ret = 0;

    if (w != cmpW)
    {
        printf("Width not match!\n");
        ret = -1;
    }

    if (h != cmpH)
    {
        printf("Height not match!\n");
        ret = -1;
    }

    if (ch != cmpCh)
    {
        printf("Channel not match!\n");
        ret = -1;
    }

    return ret;
}

int main()
{
    int w, h, ch;
    int cmpW, cmpH, cmpCh;

    cnn_config_t cfg;
    cnn_t cnn;

    // Test #1
    test(cnn_config_create(&cfg));
    test(cnn_create(&cnn, cfg));

    cnn_config_get_input_size(cfg, &w, &h, &ch);
    cnn_get_input_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_get_output_size(cfg, &w, &h, &ch);
    cnn_get_output_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_delete(cfg);
    cnn_delete(cnn);

    // Test #2
    test(cnn_config_create(&cfg));
    test(cnn_config_append_full_connect(cfg, 16));
    test(cnn_create(&cnn, cfg));

    cnn_config_get_input_size(cfg, &w, &h, &ch);
    cnn_get_input_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_get_output_size(cfg, &w, &h, &ch);
    cnn_get_output_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_delete(cfg);
    cnn_delete(cnn);

    // Test #3
    test(cnn_config_create(&cfg));
    test(cnn_config_append_activation(cfg, CNN_SOFTMAX));
    test(cnn_create(&cnn, cfg));

    cnn_config_get_input_size(cfg, &w, &h, &ch);
    cnn_get_input_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_get_output_size(cfg, &w, &h, &ch);
    cnn_get_output_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_delete(cfg);
    cnn_delete(cnn);

    // Test #4
    test(cnn_config_create(&cfg));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, CNN_DIM_2D, 7, 3));
    test(cnn_create(&cnn, cfg));

    cnn_config_get_input_size(cfg, &w, &h, &ch);
    cnn_get_input_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_get_output_size(cfg, &w, &h, &ch);
    cnn_get_output_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_delete(cfg);
    cnn_delete(cnn);

    // Test #5
    test(cnn_config_create(&cfg));
    test(cnn_config_append_pooling(cfg, CNN_DIM_2D, CNN_POOL_MAX, 2));
    test(cnn_create(&cnn, cfg));

    cnn_config_get_input_size(cfg, &w, &h, &ch);
    cnn_get_input_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_get_output_size(cfg, &w, &h, &ch);
    cnn_get_output_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_delete(cfg);
    cnn_delete(cnn);

    // Test #6
    test(cnn_config_create(&cfg));
    test(cnn_config_append_dropout(cfg, 0.5));
    test(cnn_create(&cnn, cfg));

    cnn_config_get_input_size(cfg, &w, &h, &ch);
    cnn_get_input_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_get_output_size(cfg, &w, &h, &ch);
    cnn_get_output_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_delete(cfg);
    cnn_delete(cnn);

    // Test #7
    test(cnn_config_create(&cfg));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, CNN_DIM_2D, 3, 7));
    test(cnn_config_append_pooling(cfg, CNN_DIM_2D, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, CNN_DIM_2D, 16, 7));
    test(cnn_config_append_pooling(cfg, CNN_DIM_2D, CNN_POOL_MAX, 2));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_full_connect(cfg, 128));
    test(cnn_config_append_dropout(cfg, 0.5));
    test(cnn_config_append_full_connect(cfg, 64));
    test(cnn_config_append_dropout(cfg, 0.5));
    test(cnn_config_append_full_connect(cfg, 10));
    test(cnn_config_append_activation(cfg, CNN_SOFTMAX));
    test(cnn_create(&cnn, cfg));

    cnn_config_get_input_size(cfg, &w, &h, &ch);
    cnn_get_input_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_get_output_size(cfg, &w, &h, &ch);
    cnn_get_output_size(cnn, &cmpW, &cmpH, &cmpCh);
    test(size_cmp(w, h, ch, cmpW, cmpH, cmpCh));

    cnn_config_delete(cfg);
    cnn_delete(cnn);

    return 0;
}
