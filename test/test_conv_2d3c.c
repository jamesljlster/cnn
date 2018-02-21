#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cnn.h>
#include <cnn_private.h>
#include <cnn_calc.h>

#define KERNEL_SIZE 3
#define CHANNEL 3
#define IMG_WIDTH 4
#define IMG_HEIGHT 4

#define test(func) \
	ret = func; \
	if(ret < 0) \
	{ \
		printf("%s failed with error: %d\n", #func, ret); \
		return -1; \
	}

void print_img(float* src, int rows, int cols, int channel)
{
	int i, j, k;

	printf("[\n");
	for(i = 0; i < rows; i++)
	{
		printf("[ ");
		for(j = 0; j < cols; j++)
		{
			printf("(");
			for(k = 0; k < channel; k++)
			{
				printf("%g", src[i * cols + j + k]);
				if(k < channel - 1)
				{
					printf(", ");
				}
				else
				{
					printf(")");
				}
			}

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
	int i;
	int ret;
	union CNN_LAYER layer[3];

	float src[IMG_WIDTH * IMG_HEIGHT * CHANNEL];
	float kernel[CHANNEL * KERNEL_SIZE * KERNEL_SIZE];
	float bias[(IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1)];
	float desire[(IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1)];

	cnn_config_t cfg = NULL;

	// Create cnn
	test(cnn_config_create(&cfg));
	test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, CHANNEL));
	test(cnn_config_set_layers(cfg, 3));
	test(cnn_config_set_activation(cfg, 1, CNN_RELU));
	test(cnn_config_set_convolution(cfg, 2, 2, 3));

	// Rand
	srand(time(NULL));
	for(i = 0; i < IMG_WIDTH * IMG_HEIGHT * CHANNEL; i++)
	{
		src[i] = rand() % 5;
	}

	for(i = 0; i < CHANNEL * KERNEL_SIZE * KERNEL_SIZE; i++)
	{
		kernel[i] = rand() % 5;
	}

	for(i = 0; i < (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1); i++)
	{
		bias[i] = rand() % 5;
		desire[i] = rand() % 5;
	}

	// Print information
	printf("src:\n");
	print_img(src, IMG_HEIGHT, IMG_WIDTH, CHANNEL);
	printf("\n");

	printf("kernel:\n");
	print_img(kernel, CHANNEL * KERNEL_SIZE, KERNEL_SIZE, 1);
	printf("\n");

	printf("bias:\n");
	print_img(bias, 1, (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1), 1);
	printf("\n");

	printf("desire:\n");
	print_img(desire, 1, (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1), 1);
	printf("\n");

	// Allocate cnn layer
	test(cnn_layer_afunc_alloc(&layer[1].aFunc, IMG_WIDTH, IMG_HEIGHT, CHANNEL, 1, CNN_RELU));
	test(cnn_layer_conv_alloc(&layer[2].conv, IMG_WIDTH, IMG_HEIGHT, CHANNEL, KERNEL_SIZE, 1));

	// Copy memory
	memcpy(layer[1].outMat.data.mat, src, sizeof(float) *
			layer[1].outMat.data.rows * layer[1].outMat.data.cols);
	memcpy(layer[2].conv.kernel.mat, kernel, sizeof(float) *
			layer[2].conv.kernel.rows * layer[2].conv.kernel.cols);
	memcpy(layer[2].conv.bias.mat, bias, sizeof(float) *
			layer[2].conv.bias.rows * layer[2].conv.bias.cols);

	// Forward
	printf("***** Forward *****\n");
	cnn_forward_conv(layer, cfg, 2);

	// Print detail
	printf("Convolution output:\n");
	print_img(layer[2].outMat.data.mat, layer[2].outMat.height, layer[2].outMat.width,
			layer[2].outMat.channel);

	return 0;
}
