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
#define BATCH 2

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
	int wRows, wCols;

	cnn_config_t cfg = NULL;
	cnn_t cnn = NULL;

	float kernel[KERNEL_SIZE * KERNEL_SIZE] = {
		-0.4, -0.3, -0.2,
		-0.1,  0.0,  0.1,
		 0.2,  0.3,  0.4
	};

	float weight[(IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_HEIGHT - KERNEL_SIZE + 1) * OUTPUTS];
	float bias[OUTPUTS];

	float src[IMG_WIDTH * IMG_HEIGHT * BATCH];
	float output[OUTPUTS * BATCH];
	float desire[OUTPUTS * BATCH] = {
		1, 0, 0,
		0, 1, 0
	};

	float err[OUTPUTS * BATCH];

	// Set input image
	for(i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
	{
		src[i] = i;
	}

	for(i = 0; i < IMG_HEIGHT; i++)
	{
		for(j = 0; j < IMG_WIDTH; j++)
		{
			src[(i * IMG_WIDTH + j) + IMG_WIDTH * IMG_HEIGHT] = i;
		}
	}

	// Set weight
	wRows = (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_HEIGHT - KERNEL_SIZE + 1);
	wCols = OUTPUTS;
	for(i = 0; i < wRows * wCols; i++)
	{
		weight[i] = (float)i / 10.0;
	}

	// Set bias
	for(i = 0; i < OUTPUTS; i++)
	{
		bias[i] = (float)i / 10.0 + 0.5;
	}

	// Set config
	test(cnn_config_create(&cfg));
	test(cnn_config_set_batch_size(cfg, BATCH));
	test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, 1));
	test(cnn_config_set_layers(cfg, 5));
	test(cnn_config_set_convolution(cfg, 1, 2, 3));
	test(cnn_config_set_activation(cfg, 2, CNN_RELU));
	test(cnn_config_set_full_connect(cfg, 3, OUTPUTS));
	test(cnn_config_set_activation(cfg, 4, CNN_SOFTMAX));

	// Create cnn
	test(cnn_create(&cnn, cfg));

	// Set kernel
	memcpy(cnn->layerList[1].conv.kernel.mat, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

	// Set weight
	memcpy(cnn->layerList[3].fc.weight.mat, weight, wRows * wCols * sizeof(float));

	// Set bias
	memcpy(cnn->layerList[3].fc.bias.mat, bias, OUTPUTS * sizeof(float));

	// CNN forward
	cnn_forward(cnn, src, output);

	printf("=== Source image ===\n");
	print_img(src, IMG_HEIGHT, IMG_WIDTH);
	printf("\n");

	printf("=== Kernel ===\n");
	print_img(kernel, KERNEL_SIZE, KERNEL_SIZE);
	printf("\n");

	printf("=== Weight ===\n");
	print_img(weight, wRows, wCols);
	printf("\n");

	printf("=== Bias ===\n");
	print_img(bias, 1, OUTPUTS);
	printf("\n");

	printf("=== Desire ===\n");
	print_img(desire, BATCH, OUTPUTS);
	printf("\n");

	// Print detail
	printf("***** Network Detail *****\n");
	for(i = 0; i < cnn->cfg.layers; i++)
	{
		printf("=== Layer %d output ===\n", i);
		print_img(cnn->layerList[i].outMat.data.mat,
				cnn->layerList[i].outMat.data.rows,
				cnn->layerList[i].outMat.data.cols);

		printf("\n");
	}

	// Find error
	for(i = 0; i < OUTPUTS * BATCH; i++)
	{
		err[i] = desire[i] - output[i];
	}

	// CNN Backpropagation
	cnn_backward(cnn, err);

	// Print detail
	printf("***** Network Gradient Detail *****\n");
	for(i = cnn->cfg.layers - 1; i > 0; i--)
	{
		printf("=== Layer %d ===\n", i);
		printf("Gradient:\n");
		print_img(cnn->layerList[i].outMat.data.grad,
				cnn->layerList[i].outMat.data.rows,
				cnn->layerList[i].outMat.data.cols);
		printf("\n");
	}

	// Print updated network
	printf("***** Updated Network Detail *****\n");
	for(i = cnn->cfg.layers - 1; i >= 0; i--)
	{
		printf("=== Layer %d ===\n", i);
		switch(cnn->cfg.layerCfg[i].type)
		{
			case CNN_LAYER_FC:
				printf("- Fully Connected -\n");
				printf("Weight:\n");
				print_img(cnn->layerList[i].fc.weight.mat,
						cnn->layerList[i].fc.weight.rows,
						cnn->layerList[i].fc.weight.cols);
				printf("\n");
				printf("Bias:\n");
				print_img(cnn->layerList[i].fc.bias.mat,
						cnn->layerList[i].fc.bias.rows,
						cnn->layerList[i].fc.bias.cols);
				printf("\n");
				break;

			case CNN_LAYER_CONV:
				printf("- Convolution -\n");
				printf("Kernel:\n");
				print_img(cnn->layerList[i].conv.kernel.mat,
						cnn->layerList[i].conv.kernel.rows,
						cnn->layerList[i].conv.kernel.cols);
				printf("\n");
				printf("Bias:\n");
				print_img(cnn->layerList[i].conv.bias.mat,
						cnn->layerList[i].conv.bias.rows,
						cnn->layerList[i].conv.bias.cols);
				printf("\n");
				break;

			case CNN_LAYER_AFUNC:
				printf("- Activation Function -\n");
				break;
		}
		printf("\n");
	}

	return 0;
}
