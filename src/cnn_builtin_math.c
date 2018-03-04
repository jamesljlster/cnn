#include <string.h>
#include <math.h>

#include "cnn.h"
#include "cnn_builtin_math.h"

CNN_ACTIV_DEF((*cnn_activ_list[])) = {
	cnn_softmax,
	cnn_relu,
	cnn_swish
};

CNN_ACTIV_DEF((*cnn_activ_grad_list[])) = {
	cnn_softmax_grad,
	cnn_relu_grad,
	cnn_swish_grad
};

const char* cnn_activ_name[] = {
	"Softmax",
	"ReLU",
	"Swish"
};

CNN_ACTIV_DEF(cnn_softmax)
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

CNN_ACTIV_DEF(cnn_softmax_grad)
{
	int i, j;

	// Find softmax gradient
	cnn_softmax(buf, src, len, NULL);
	for(i = 0; i < len; i++)
	{
		for(j = 0; j < len; j++)
		{
			dst[i * len + j] = buf[i] * ((float)(i == j) - buf[j]);
		}
	}
}

CNN_ACTIV_DEF(cnn_relu)
{
	int i;
	for(i = 0; i < len; i++)
	{
		dst[i] = fmaxf(src[i], 0.0f);
	}
}

CNN_ACTIV_DEF(cnn_relu_grad)
{
	int i;

	// Find relu gradient
	memset(dst, 0, len * sizeof(float));
	for(i = 0; i < len; i++)
	{
		dst[i] = (src[i] < 0.0f) ? 0 : 1;
	}
}

CNN_ACTIV_DEF(cnn_swish)
{
	int i;
	for(i = 0; i < len; i++)
	{
		dst[i] = src[i] / (1.0f + expf(-src[i]));
	}
}

CNN_ACTIV_DEF(cnn_swish_grad)
{
	int i;

	// Find swish gradient
	memset(dst, 0, len * sizeof(float));
	cnn_swish(buf, src, len, NULL);
	for(i = 0; i < len; i++)
	{
		dst[i] = buf[i] + (buf[i] / src[i]) * (1.0f - buf[i]);
	}
}

int cnn_get_activ_id(const char* name)
{
	int i;
	int ret = CNN_PARSE_FAILED;

	if(name != NULL)
	{
		for(i = 0; i < CNN_ACTIV_AMOUNT; i++)
		{
			ret = strcmp(name, cnn_activ_name[i]);
			if(ret == 0)
			{
				ret = i;
				goto RET;
			}
		}
	}

RET:
	return ret;
}
