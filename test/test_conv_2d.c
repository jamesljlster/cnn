#include <stdio.h>

#include <cnn_calc.h>

#define KERNEL_SIZE 3
#define IMG_WIDTH 5
#define IMG_HEIGHT 5

#define DST_WIDTH 3
#define DST_HEIGHT 3

void print_img(float* src, int rows, int cols)
{
	int i, j;

	for(i = 0; i < rows; i++)
	{
		for(j = 0; j < cols; j++)
		{
			printf("%+5.2f", src[i * cols + j]);
			if(j < cols - 1)
			{
				printf("  ");
			}
			else
			{
				printf("\n");
			}
		}
	}
}

int main()
{
	int i;
	float kernel[KERNEL_SIZE * KERNEL_SIZE] = {
		-4, -3, -2,
		-1,  0,  1,
		 2,  3,  4
	};

	float src[IMG_WIDTH * IMG_HEIGHT];
	float dst[DST_WIDTH * DST_HEIGHT];

	for(i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
	{
		src[i] = i;
	}

	cnn_conv_2d(dst, DST_HEIGHT, DST_WIDTH, kernel, KERNEL_SIZE, src, IMG_HEIGHT, IMG_WIDTH);

	printf("kernel:\n");
	print_img(kernel, KERNEL_SIZE, KERNEL_SIZE);
	printf("\n");

	printf("src:\n");
	print_img(src, IMG_HEIGHT, IMG_WIDTH);
	printf("\n");

	printf("dst:\n");
	print_img(dst, DST_HEIGHT, DST_WIDTH);
	printf("\n");

	return 0;
}
