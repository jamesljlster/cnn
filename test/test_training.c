#include <stdio.h>
#include <string.h>

#include <cnn.h>
#include <cnn_private.h>
#include <cnn_calc.h>

#define KERNEL_SIZE 3
#define IMG_WIDTH 5
#define IMG_HEIGHT 5

#define DST_WIDTH 3
#define DST_HEIGHT 3

#define OUTPUTS 3
#define ITER 10000

#define test(func) \
	ret = func; \
	if(ret < 0) \
	{ \
		printf("%s failed with error: %d\n", #func, ret); \
		return -1; \
	}

void print_img(float* src, int rows, int cols)
{
	int i, j;

	printf("[\n");
	for(i = 0; i < rows; i++)
	{
		printf("[ ");
		for(j = 0; j < cols; j++)
		{
			printf("%+5e", src[i * cols + j]);
			if(j < cols - 1)
			{
				printf(", ");
			}
			else
			{
				if(i < rows - 1)
				{
					printf(" ],\n");
				}
				else
				{
					printf(" ]\n");
				}
			}
		}
	}
	printf("]\n");
}

int main()
{
	int i, j;
	int ret;
	float mse;

	cnn_config_t cfg = NULL;
	cnn_t cnn = NULL;

	float src[IMG_WIDTH * IMG_HEIGHT];
	float output[OUTPUTS];
	float desire[OUTPUTS] = {1, 0, 0};
	float err[OUTPUTS];

	// Set input image
	for(i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
	{
		src[i] = i;
	}

	// Set config
	test(cnn_config_create(&cfg));
	test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT));
	test(cnn_config_set_layers(cfg, 5));
	test(cnn_config_set_convolution(cfg, 1, 2, 3));
	test(cnn_config_set_activation(cfg, 2, CNN_RELU));
	test(cnn_config_set_full_connect(cfg, 3, OUTPUTS));
	test(cnn_config_set_activation(cfg, 4, CNN_SOFTMAX));

	// Create cnn
	test(cnn_create(&cnn, cfg));

	// Training
	for(i = 0; i < ITER; i++)
	{
		test(cnn_training_custom(cnn, 0.01, src, desire, output, err));

		mse = 0;
		for(j = 0; j < OUTPUTS; j++)
		{
			mse += err[j] * err[j];
		}
		mse /= (float)OUTPUTS;
		printf("Iter %d, mse: %f\n", i, mse);
	}

	return 0;
}
