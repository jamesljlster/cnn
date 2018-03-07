#include <stdio.h>
#include <string.h>
#include <time.h>

#include <cnn.h>
#include <cnn_private.h>
#include <cnn_calc.h>

#define KERNEL_SIZE 3
#define IMG_WIDTH 5
#define IMG_HEIGHT 5

#define DST_WIDTH 3
#define DST_HEIGHT 3

#define OUTPUTS 3

#define ITER 100000

#define test(func) \
	ret = func; \
	if(ret < 0) \
	{ \
		printf("%s failed with error: %d\n", #func, ret); \
		return -1; \
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
	test(cnn_config_set_layers(cfg, 8));

	test(cnn_config_set_convolution(cfg, 1, 2, 1, 3));
	test(cnn_config_set_activation(cfg, 2, CNN_RELU));
	test(cnn_config_set_convolution(cfg, 3, 2, 1, 3));
	test(cnn_config_set_activation(cfg, 4, CNN_RELU));
	test(cnn_config_set_full_connect(cfg, 5, 1024));
	test(cnn_config_set_full_connect(cfg, 6, OUTPUTS));
	test(cnn_config_set_activation(cfg, 7, CNN_SOFTMAX));

	// Create cnn
	test(cnn_create(&cnn, cfg));

	// Hold current time
#ifdef _MSC_VER
	timeHold = clock();
#else
	clock_gettime(CLOCK_MONOTONIC, &timeHold);
#endif

	// CNN Backpropagation
	for(i = 0; i < ITER; i++)
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
