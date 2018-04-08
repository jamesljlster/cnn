#include <stdio.h>
#include <string.h>
#include <time.h>

#include <cnn.h>
#include <cnn_private.h>
#include <cnn_calc.h>

#include "test.h"

#define WIDTH 64
#define HEIGHT 64
#define CH_IN 3
#define ITER 10000

int main(int argc, char* argv[])
{
	int i;

	cnn_config_t cfg = NULL;
	cnn_t cnn = NULL;

#ifdef _MSC_VER
	clock_t timeHold, tmpTime;
#else
	struct timespec timeHold, tmpTime;
#endif
	float timeCost;

	int kSize;
	int fSize;
	int width = WIDTH;
	int height = HEIGHT;
	int chIn = CH_IN;
	int iter = ITER;

	int wOut;
	int hOut;
	int chOut;

	float* in = NULL;
	float* out = NULL;

	// Checking arguments
	if(argc < 3)
	{
		printf("Usage: test_conv_perf <kernel_size> <filter_size> "
				"<input_width=%d> <input_height=%d> <input_channel=%d> "
				"<iteration=%d>\n", WIDTH, HEIGHT, CH_IN, ITER);
		return -1;
	}

	// Parse arguments
	kSize = atoi(argv[1]);
	fSize = atoi(argv[2]);

	if(argc > 3)
	{
		width = atoi(argv[3]);
	}

	if(argc > 4)
	{
		height = atoi(argv[4]);
	}

	if(argc > 5)
	{
		chIn = atoi(argv[5]);
	}

	if(argc > 6)
	{
		iter = atoi(argv[6]);
	}

	printf("Using:\n");
	printf("kernel_size: %d\n", kSize);
	printf("filter_size: %d\n", fSize);
	printf("input_width: %d\n", width);
	printf("input_height: %d\n", height);
	printf("input_channel: %d\n", chIn);
	printf("iter: %d\n", iter);
	printf("\n");

	// Set config
	test(cnn_config_create(&cfg));
	test(cnn_config_set_input_size(cfg, width, height, chIn));

	test(cnn_config_append_activation(cfg, CNN_RELU));
	test(cnn_config_append_convolution(cfg, 2, fSize, kSize));

	// Create cnn
	test(cnn_create(&cnn, cfg));
	cnn_get_output_size(cnn, &wOut, &hOut, &chOut);
	alloc(in, width * height * chIn, float);
	alloc(out, wOut * hOut * chOut, float);

	// Hold current time
#ifdef _MSC_VER
	timeHold = clock();
#else
	clock_gettime(CLOCK_MONOTONIC, &timeHold);
#endif

	// forward and backward
	for(i = 0; i < iter; i++)
	{
		cnn_forward(cnn, in, out);
		cnn_backward(cnn, out);
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
	printf("With %d iteration, time cost: %f ms\n", iter, timeCost);

	return 0;
}
