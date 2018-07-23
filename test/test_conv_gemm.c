#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cnn.h>
#include <cnn_private.h>
#include <cnn_calc.h>

#include <cblas.h>

#include "test.h"

#define KERNEL_SIZE 3
#define CH_IN 3
#define CH_OUT 2
#define IMG_WIDTH 4
#define IMG_HEIGHT 4

void print_img(float* src, int width, int height, int channel);
void print_img_int(int* src, int width, int height, int channel);

/*
void cnn_conv_unroll_2d(int* indexMap, int dstHeight, int dstWidth, int kSize,
		int srcHeight, int srcWidth, int srcCh)
{
	int __kMemSize = kSize * kSize;
	int __srcImSize = srcHeight * srcWidth;
	int __indexMapCols = __kMemSize * srcCh;

	for(int __h = 0; __h < dstHeight; __h++)
	{
		int __dstRowShift = __h * dstHeight;

		for(int __w = 0; __w < dstWidth; __w++)
		{
			int __indexMapRow = __dstRowShift + __w;
			int __indexMemBase = __indexMapRow * __indexMapCols;

			for(int __ch = 0; __ch < srcCh; __ch++)
			{
				int __indexMemShiftBase = __indexMemBase + __kMemSize * __ch;
				int __srcChShift = __ch * __srcImSize;

				for(int __convH = 0; __convH < kSize; __convH++)
				{
					int __indexMemShift = __indexMemShiftBase + __convH * kSize;
					int __srcShift = (__h + __convH) * srcWidth + __srcChShift;

					for(int __convW = 0; __convW < kSize; __convW++)
					{
						indexMap[__indexMemShift + __convW] = __srcShift +
							(__w + __convW);
					}
				}
			}
		}
	}
}
*/

int main()
{
	//int i;
	union CNN_LAYER layer[3];

	//float src[IMG_WIDTH * IMG_HEIGHT * CH_IN] = {0};
	float src[IMG_WIDTH * IMG_HEIGHT * CH_IN] = {
		1, 0, 3, 0,
		0, 3, 0, 4,
		4, 0, 3, 0,
		3, 1, 0, 4,

		2, 3, 2, 2,
		0, 3, 3, 0,
		0, 2, 2, 4,
		2, 1, 4, 0,

		4, 0, 0, 4,
		3, 2, 0, 2,
		3, 1, 4, 3,
		4, 1, 4, 1
	};

	//float kernel[CH_OUT * CH_IN * KERNEL_SIZE * KERNEL_SIZE] = {0};
	float kernel[CH_OUT * CH_IN * KERNEL_SIZE * KERNEL_SIZE] = {
		0, 3, 4, 0, 1, 2, 2, 4, 1,
		1, 3, 0, 0, 2, 0, 4, 4, 2,
		0, 4, 4, 1, 3, 4, 2, 4, 4,

		3, 1, 0, 4, 4, 3, 0, 1, 1,
		3, 2, 3, 0, 4, 2, 1, 2, 0,
		2, 0, 2, 1, 0, 3, 1, 4, 4
	};

	//float bias[CH_OUT * (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1)] = {0};
	float bias[CH_OUT * (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1)] = {
		3, 1, 3, 4,
		0, 4, 1, 1
	};

	//float desire[CH_OUT * (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1)] = {0};
	float desire[CH_OUT * (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1)] = {
		3, 1, 4, 2,
		4, 4, 4, 4
	};

	int dstWidth, dstHeight;
	int indexMapRows, indexMapCols;
	int* indexMap;
	float* unrollImg;
	float* unrollImgGrad;

	cnn_config_t cfg = NULL;

	// Allocate indexMap
	dstWidth = IMG_WIDTH - KERNEL_SIZE + 1;
	dstHeight = IMG_HEIGHT - KERNEL_SIZE + 1;
	indexMapRows = dstWidth * dstHeight;
	indexMapCols = CH_IN * KERNEL_SIZE * KERNEL_SIZE;

	alloc(indexMap, indexMapRows * indexMapCols, int);
	alloc(unrollImg, indexMapRows * indexMapCols, float);
	alloc(unrollImgGrad, indexMapRows * indexMapCols, float);

	cnn_conv_unroll_2d(indexMap, dstHeight, dstWidth, KERNEL_SIZE,
			IMG_HEIGHT, IMG_WIDTH, CH_IN);

	for(int i = 0; i < indexMapRows * indexMapCols; i++)
	{
		unrollImg[i] = src[indexMap[i]];
	}

	//printf("indexMap:\n");
	//print_img_int(indexMap, indexMapCols, indexMapRows, 1);
	//printf("\n");
	//printf("unrollImg:\n");
	//print_img(unrollImg, indexMapCols, indexMapRows, 1);
	//printf("\n");

	// Create cnn
	test(cnn_config_create(&cfg));
	test(cnn_config_set_input_size(cfg, IMG_WIDTH, IMG_HEIGHT, CH_IN));
	test(cnn_config_set_layers(cfg, 3));
	test(cnn_config_set_activation(cfg, 1, CNN_RELU));
	test(cnn_config_set_convolution(cfg, 2, CNN_DIM_2D, CH_OUT, KERNEL_SIZE));

	// Rand
	/*
	srand(time(NULL));
	for(i = 0; i < IMG_WIDTH * IMG_HEIGHT * CH_IN; i++)
	{
		src[i] = rand() % 5;
	}

	for(i = 0; i < CH_OUT * CH_IN * KERNEL_SIZE * KERNEL_SIZE; i++)
	{
		kernel[i] = rand() % 5;
	}

	for(i = 0; i < CH_OUT * (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1); i++)
	{
		bias[i] = rand() % 5;
		desire[i] = rand() % 5;
	}
	*/

	// Print information
	printf("src:\n");
	print_img(src, IMG_WIDTH, IMG_HEIGHT, CH_IN);
	printf("\n");

	printf("kernel:\n");
	print_img(kernel, KERNEL_SIZE, KERNEL_SIZE * CH_IN, CH_OUT);
	printf("\n");

	printf("bias:\n");
	print_img(bias, (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1), 1, CH_OUT);
	printf("\n");

	printf("desire:\n");
	print_img(desire, (IMG_WIDTH - KERNEL_SIZE + 1) * (IMG_WIDTH - KERNEL_SIZE + 1), 1, CH_OUT);
	printf("\n");

	// Allocate cnn layer
	test(cnn_layer_activ_alloc(&layer[1].activ, IMG_WIDTH, IMG_HEIGHT, CH_IN, 1, CNN_RELU));
	test(cnn_layer_conv_alloc(&layer[2].conv, IMG_WIDTH, IMG_HEIGHT, CH_IN, CH_OUT,
				KERNEL_SIZE, 1));

	// Copy memory
	memcpy(layer[1].outMat.data.mat, src, sizeof(float) *
			layer[1].outMat.data.rows * layer[1].outMat.data.cols);
	memcpy(layer[2].conv.kernel.mat, kernel, sizeof(float) *
			layer[2].conv.kernel.rows * layer[2].conv.kernel.cols);

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
	memcpy(layer[2].conv.bias.mat, bias, sizeof(float) *
			layer[2].conv.bias.rows * layer[2].conv.bias.cols);
#endif

	// Forward
	printf("***** Forward *****\n");
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			CH_OUT, indexMapRows, indexMapCols, 1.0,
			kernel, indexMapCols, unrollImg, indexMapCols,
			0.0, layer[2].outMat.data.mat, indexMapRows);

#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
	cblas_saxpy(layer[2].conv.bias.cols, 1.0,
			layer[2].conv.bias.mat, 1,
			layer[2].outMat.data.mat, 1);
#endif

	//cnn_forward_conv(layer, cfg, 2);

	printf("Convolution output:\n");
	print_img(layer[2].outMat.data.mat, layer[2].outMat.width, layer[2].outMat.height,
			layer[2].outMat.channel);
	printf("\n");

	printf("***** Forward #2 *****\n");
	cnn_forward_conv(layer, cfg, 2);

	printf("Convolution output:\n");
	print_img(layer[2].outMat.data.mat, layer[2].outMat.width, layer[2].outMat.height,
			layer[2].outMat.channel);
	printf("\n");

	// BP
	printf("***** BP *****\n");
	memcpy(layer[2].outMat.data.grad, desire, sizeof(float) *
			layer[2].outMat.data.rows * layer[2].outMat.data.cols);

	// Find kernel gradient
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			CH_OUT, indexMapCols, indexMapRows, 1.0,
			layer[2].outMat.data.grad, indexMapRows,
			unrollImg, indexMapCols, 1.0,
			layer[2].conv.kernel.grad, indexMapCols);

	// Find layer gradient
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
			indexMapRows, indexMapCols, CH_OUT, 1.0,
			layer[2].outMat.data.grad, indexMapRows,
			layer[2].conv.kernel.mat, indexMapCols, 1.0,
			unrollImgGrad, indexMapCols);

	for(int i = 0; i < indexMapRows * indexMapCols; i++)
	{
		layer[1].outMat.data.grad[indexMap[i]] += unrollImgGrad[i];
	}

	//cnn_backward_conv(layer, cfg, 2);

	printf("Convolution layer gradient:\n");
	print_img(layer[2].outMat.data.grad, layer[2].outMat.width, layer[2].outMat.height,
			layer[2].outMat.channel);
	printf("\n");

	printf("Previous layer gradient:\n");
	print_img(layer[1].outMat.data.grad, layer[1].outMat.width, layer[1].outMat.height,
			layer[1].outMat.channel);
	printf("\n");

	printf("Kernel gradient:\n");
	print_img(layer[2].conv.kernel.grad, KERNEL_SIZE, KERNEL_SIZE * CH_IN, CH_OUT);
	printf("\n");

	return 0;
}

void print_img(float* src, int width, int height, int channel)
{
	int imSize = width * height;

	for(int ch = 0; ch < channel; ch++)
	{
		int chShift = ch * imSize;

		printf("[\n");
		for(int h = 0; h < height; h++)
		{
			int shift = h * width + chShift;

			printf("[");
			for(int w = 0; w < width; w++)
			{
				printf("%g", src[shift + w]);
				if(w < width - 1)
				{
					printf(", ");
				}
				else
				{
					printf("]\n");
				}
			}
		}
		printf("]\n");
	}
}

void print_img_int(int* src, int width, int height, int channel)
{
	int imSize = width * height;

	for(int ch = 0; ch < channel; ch++)
	{
		int chShift = ch * imSize;

		printf("[\n");
		for(int h = 0; h < height; h++)
		{
			int shift = h * width + chShift;

			printf("[");
			for(int w = 0; w < width; w++)
			{
				printf("%d", src[shift + w]);
				if(w < width - 1)
				{
					printf(", ");
				}
				else
				{
					printf("]\n");
				}
			}
		}
		printf("]\n");
	}
}

