#include <stdio.h>
#include <string.h>
#include <time.h>

#include <cnn.h>
#include <cnn_calc.h>
#include <cnn_private.h>

#define KERNEL_SIZE 3
#define FILTER_SIZE 6
#define IMG_WIDTH 5
#define IMG_HEIGHT 5

#define DST_WIDTH 3
#define DST_HEIGHT 3

#define OUTPUTS 3

#define ITER 100000

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
    int ret;

    cnn_config_t cfg = NULL;
    cnn_t cnn = NULL;

    float err[OUTPUTS] = {0};

#ifdef _MSC_VER
    clock_t timeHold, tmpTime;
#else
    struct timespec timeHold, tmpTime;
#endif
    float timeCost;

    // Set config
    test(cnn_config_create(&cfg));
    test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, 1));

    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 2, FILTER_SIZE,
                                       KERNEL_SIZE));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_convolution(cfg, CNN_PAD_VALID, 2, FILTER_SIZE,
                                       KERNEL_SIZE));
    test(cnn_config_append_activation(cfg, CNN_RELU));
    test(cnn_config_append_full_connect(cfg, 1024));
    test(cnn_config_append_full_connect(cfg, OUTPUTS));
    test(cnn_config_append_activation(cfg, CNN_SOFTMAX));

    // Create cnn
    test(cnn_create(&cnn, cfg));

    // Hold current time
#ifdef _MSC_VER
    timeHold = clock();
#else
    clock_gettime(CLOCK_MONOTONIC, &timeHold);
#endif

    // CNN Backpropagation
    for (i = 0; i < ITER; i++)
    {
        cnn_backward(cnn, err);
    }

    // Calculate time cost
#ifdef _MSC_VER
    tmpTime = clock();
    timeCost = (float)(tmpTime - timeHold) * 1000.0 / (float)CLOCKS_PER_SEC;
#else
    clock_gettime(CLOCK_MONOTONIC, &tmpTime);
    timeCost = (tmpTime.tv_sec - timeHold.tv_sec) * 1000 +
               (float)(tmpTime.tv_nsec - timeHold.tv_nsec) / 1000000.0;
#endif
    printf("With %d iteration, time cost: %f ms\n", ITER, timeCost);

    return 0;
}
