#include <stdio.h>
#include <stdlib.h>
#include <cnn_builtin_math.h>

void print_vec(float* src, int len);

int main(int argc, char* argv[])
{
	int i;
	int len;

	float* src = NULL;
	float* dst = NULL;

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
	if(src == NULL || dst == NULL)
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
	cnn_softmax(dst, src, len);

	printf("src: ");
	print_vec(src, len);
	printf("dst: ");
	print_vec(dst, len);

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
