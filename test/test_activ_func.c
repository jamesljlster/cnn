#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <cnn.h>
#include <cnn_builtin_math.h>

void print_mat(float* src, int rows, int cols);

int main(int argc, char* argv[])
{
	int id;
	int i, j;
	int len;

	float* src = NULL;
	float* dst = NULL;
	float* buf = NULL;
	float* deri = NULL;
	float* grad = NULL;

	float err;
	float dx = pow(10, -4);

	// Check dx
	assert(dx != 0.0f);

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

	// Test activation functions
	for(id = 0; id < CNN_AFUNC_AMOUNT; id++)
	{
		// Parse argument
		for(i = 0; i < len; i++)
		{
			src[i] = atof(argv[i + 1]);
		}

		// Find grad
		memset(grad, 0, len * len * sizeof(float));
		cnn_afunc_list[id](dst, src, len, NULL);
		if(id == CNN_SOFTMAX)
		{
			for(i = 0; i < len; i++)
			{
				for(j = 0; j < len; j++)
				{
					src[j] = atof(argv[j + 1]);
					if(i == j)
					{
						src[j] += dx;
					}
				}

				cnn_afunc_list[id](buf, src, len, NULL);

				for(j = 0; j < len; j++)
				{
					grad[i * len + j] = (buf[j] - dst[j]) / dx;
				}
			}
		}
		else
		{
			for(i = 0; i < len; i++)
			{
				for(j = 0; j < len; j++)
				{
					src[j] = atof(argv[j + 1]);
					if(i == j)
					{
						src[j] += dx;
					}
				}

				cnn_afunc_list[id](buf, src, len, NULL);

				grad[i] = (buf[i] - dst[i]) / dx;
			}
		}

		// Find derivative
		for(i = 0; i < len; i++)
		{
			src[i] = atof(argv[i + 1]);
		}

		memset(deri, 0, len * len * sizeof(float));
		cnn_afunc_grad_list[id](deri, src, len, buf);

		// Find error
		err = 0;
		for(i = 0; i < len * len; i++)
		{
			err += fabs(grad[i] - deri[i]);
		}

		printf("=== Test %s derivative ===\n", cnn_afunc_name[id]);
		printf("deri:\n");
		print_mat(deri, len, len);
		printf("\n");
		printf("grad:\n");
		print_mat(grad, len, len);
		printf("\n");
		printf("Sum of error: %lf\n", err);
		printf("\n");
	}

	return 0;
}

void print_mat(float* src, int rows, int cols)
{
	int i, j;
	for(i = 0; i < rows; i++)
	{
		printf(" | ");
		for(j = 0; j < cols; j++)
		{
			printf("%+f", src[i * cols + j]);
			if(j < cols - 1)
			{
				printf("  ");
			}
		}
		printf(" |\n");
	}
}
