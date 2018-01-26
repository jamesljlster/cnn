#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cnn_builtin_math.h>

#define DX 0.00001

void print_vec(float* src, int len);

int main(int argc, char* argv[])
{
	int i, j;
	int len;

	float* src = NULL;
	float* dst = NULL;
	float* buf = NULL;
	float* deri = NULL;
	float* grad = NULL;

	float err;

	// Check argument
	if(argc <= 1)
	{
		printf("Assign arguments with real numbers to run the program\n");
		return -1;
	}

	// Memory allocation
	len = argc - 1;
	src = calloc(len, sizeof(float));
	dst = calloc(len, sizeof(float));
	buf = calloc(len, sizeof(float));
	deri = calloc(len * len, sizeof(float));
	grad = calloc(len * len, sizeof(float));
	if(src == NULL || dst == NULL || buf == NULL || deri == NULL || grad == NULL)
	{
		printf("Memory allocation failed!\n");
		return -1;
	}

	// Parse argument
	for(i = 0; i < len; i++)
	{
		src[i] = atof(argv[i + 1]);
	}

	// Run softmax
	cnn_softmax(dst, src, len, NULL);

	printf("Test softmax:\n");
	printf("src: ");
	print_vec(src, len);
	printf("dst: ");
	print_vec(dst, len);
	printf("\n");

	// Find grad
	for(i = 0; i < len; i++)
	{
		for(j = 0; j < len; j++)
		{
			src[j] = atof(argv[j + 1]);
			if(i == j)
			{
				src[j] += DX;
			}
		}

		cnn_softmax(deri, src, len, NULL);
		grad[i] = deri[i];
	}

	for(i = 0; i < len; i++)
	{
		grad[i] = (grad[i] - dst[i]) / DX;
	}

	// Find derivative
	for(i = 0; i < len; i++)
	{
		src[i] = atof(argv[i + 1]);
	}

	cnn_softmax_grad(deri, src, len, buf);

	// Find error
	err = 0;
	for(i = 0; i < len; i++)
	{
		err += fabs(grad[i] - deri[i]);
	}

	printf("Test softmax derivative:\n");
	printf("deri: ");
	print_vec(deri, len);
	printf("grad: ");
	print_vec(grad, len);
	printf("Sum error: %lf\n", err);
	printf("\n");

	return 0;
}

void print_vec(float* src, int len)
{
	int i;
	for(i = 0; i < len; i++)
	{
		printf("%f", src[i]);
		if(i < len - 1)
		{
			printf(", ");
		}
		else
		{
			printf("\n");
		}
	}
}
