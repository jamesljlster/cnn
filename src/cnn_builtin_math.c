#include <math.h>

#include "cnn_builtin_math.h"

CNN_AFUNC_DEF(cnn_softmax)
{
	int i;
	float sum = 0;

	for(i = 0; i < len; i++)
	{
		sum += exp(src[i]);
	}

	for(i = 0; i < len; i++)
	{
		dst[i] = exp(src[i]) / sum;
	}
}

CNN_AFUNC_DEF(cnn_relu)
{

}
