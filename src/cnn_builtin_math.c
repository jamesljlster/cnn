#include <math.h>

#include "cnn.h"
#include "cnn_builtin_math.h"

CNN_AFUNC_DEF((*cnn_transfer_list[])) = {
	cnn_softmax,
	cnn_relu,
	cnn_swish
};

CNN_AFUNC_DEF((*cnn_transfer_derivative_list[])) = {
	cnn_softmax_derivative,
	cnn_relu_derivative,
	cnn_swish_derivative
};

const char* cnn_transfer_func_name[] = {
	"Softmax",
	"ReLU",
	"Swish"
};

CNN_AFUNC_DEF(cnn_softmax)
{
	int i;
	float max, sum;

	// Find max value
	max = src[0];
	for(i = 1; i < len; i++)
	{
		if(src[i] > max)
		{
			max = src[i];
		}
	}

	// Find exponential summation
	sum = 0;
	for(i = 0; i < len; i++)
	{
		dst[i] = src[i] - max;
		sum += exp(dst[i]);
	}

	// Find softmax output
	for(i = 0; i < len; i++)
	{
		dst[i] = exp(dst[i]) / sum;
	}
}

CNN_AFUNC_DEF(cnn_softmax_derivative)
{

}

CNN_AFUNC_DEF(cnn_relu)
{

}

CNN_AFUNC_DEF(cnn_relu_derivative)
{

}

CNN_AFUNC_DEF(cnn_swish)
{

}

CNN_AFUNC_DEF(cnn_swish_derivative)
{

}

