#include <stdio.h>

#include <cnn_calc.h>

#define IMG_WIDTH 7
#define IMG_HEIGHT 7

#define DST_WIDTH 3
#define DST_HEIGHT 3

#define POOL_SIZE 2

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

void print_img_int(int* src, int rows, int cols)
{
	int i, j;

	for(i = 0; i < rows; i++)
	{
		for(j = 0; j < cols; j++)
		{
			printf("%d", src[i * cols + j]);
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

	float src[IMG_WIDTH * IMG_HEIGHT];
	int index[DST_WIDTH * DST_HEIGHT];
	float dst[DST_WIDTH * DST_HEIGHT];

	for(i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
	{
		src[i] = i;
	}

	cnn_pool_2d_max(dst, index, DST_HEIGHT, DST_WIDTH, POOL_SIZE, src, IMG_HEIGHT, IMG_WIDTH);

	printf("src:\n");
	print_img(src, IMG_HEIGHT, IMG_WIDTH);
	printf("\n");

	printf("dst:\n");
	print_img(dst, DST_HEIGHT, DST_WIDTH);
	printf("\n");

	printf("index:\n");
	print_img_int(index, DST_HEIGHT, DST_WIDTH);
	printf("\n");

	return 0;
}
