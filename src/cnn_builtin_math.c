#include <math.h>
#include <string.h>

#include "cnn.h"
#include "cnn_builtin_math.h"
#include "cnn_config.h"

#ifdef CNN_WITH_CUDA
#include <cuda_runtime.h>
#include "cnn_builtin_math_cu.h"
#endif

CNN_ACTIV_DEF((*cnn_activ_list[])) = {
    cnn_softmax,        //
    cnn_relu,           //
    cnn_swish,          //
    cnn_sigmoid,        //
    cnn_tanh,           //
    cnn_gaussian,       //
    cnn_bent_identity,  //
    cnn_softplus,       //
    cnn_softsign,       //
    cnn_sinc,           //
    cnn_sinusoid        //
};

CNN_ACTIV_DEF((*cnn_activ_grad_list[])) = {
    cnn_softmax_grad,        //
    cnn_relu_grad,           //
    cnn_swish_grad,          //
    cnn_sigmoid_grad,        //
    cnn_tanh_grad,           //
    cnn_gaussian_grad,       //
    cnn_bent_identity_grad,  //
    cnn_softplus_grad,       //
    cnn_softsign_grad,       //
    cnn_sinc_grad,           //
    cnn_sinusoid_grad        //
};

const char* cnn_activ_name[] = {
    "Softmax",             //
    "ReLU",                //
    "Swish",               //
    "Sigmoid",             //
    "Hyperbolic Tangent",  //
    "Gaussian",            //
    "Bent Identity",       //
    "SoftPlus",            //
    "SoftSign",            //
    "Sinc",                //
    "Sinusoid"             //
};

CNN_ACTIV_DEF(cnn_softmax)
{
    float max, sum;

#ifdef CNN_WITH_CUDA
    // Find max value
    cnn_max_gpu(&max, src, len, buf);

    // Find shifted vector
    cnn_add_gpu(buf, src, len, -max);

    // Find exponential vector
    cnn_exp_gpu(buf, buf, len);

    // Find sum
    cnn_sum_gpu(&sum, buf, len, dst);

    // Find softmax
    cnn_div_gpu(dst, buf, len, sum);

    cudaDeviceSynchronize();
#else
    int i;

    // Find max value
    max = src[0];
    for (i = 1; i < len; i++)
    {
        if (src[i] > max)
        {
            max = src[i];
        }
    }

    // Find exponential summation
    sum = 0;
    for (i = 0; i < len; i++)
    {
        dst[i] = src[i] - max;
        sum += exp(dst[i]);
    }

    // Find softmax output
    for (i = 0; i < len; i++)
    {
        dst[i] = exp(dst[i]) / sum;
    }
#endif
}

CNN_ACTIV_DEF(cnn_softmax_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_smax_grad_gpu(dst, buf, len);
    cudaDeviceSynchronize();
#else
    int i, j;

    // Find softmax gradient
    // cnn_softmax(buf, src, len, NULL);
    for (i = 0; i < len; i++)
    {
        for (j = 0; j < len; j++)
        {
            dst[i * len + j] = buf[i] * ((float)(i == j) - buf[j]);
        }
    }
#endif
}

CNN_ACTIV_DEF(cnn_relu)
{
#ifdef CNN_WITH_CUDA
    cnn_fmaxf_gpu(dst, src, len, 0.0f);
    cudaDeviceSynchronize();
#else
    int i;
    for (i = 0; i < len; i++)
    {
        dst[i] = fmaxf(src[i], 0.0f);
    }
#endif
}

CNN_ACTIV_DEF(cnn_relu_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_relu_grad_gpu(dst, src, len);
    cudaDeviceSynchronize();
#else
    int i;

    // Find relu gradient
    // memset(dst, 0, len * sizeof(float));
    for (i = 0; i < len; i++)
    {
        dst[i] = (src[i] < 0.0f) ? 0 : 1;
    }
#endif
}

CNN_ACTIV_DEF(cnn_swish)
{
#ifdef CNN_WITH_CUDA
    cnn_swish_gpu(dst, src, len);
    cudaDeviceSynchronize();
#else
    int i;
    for (i = 0; i < len; i++)
    {
        dst[i] = src[i] / (1.0f + expf(-src[i]));
    }
#endif
}

CNN_ACTIV_DEF(cnn_swish_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_swish_grad_gpu(dst, src, buf, len);
    cudaDeviceSynchronize();
#else
    int i;

    // Find swish gradient
    // memset(dst, 0, len * sizeof(float));
    cnn_swish(buf, src, len, NULL);
    for (i = 0; i < len; i++)
    {
        if (src[i] == 0.0f)
        {
            dst[i] = 0.5;
        }
        else
        {
            dst[i] = buf[i] + (buf[i] / src[i]) * (1.0f - buf[i]);
        }
    }
#endif
}

CNN_ACTIV_DEF(cnn_sigmoid)
{
    int i;
    for (i = 0; i < len; i++)
    {
        dst[i] = 1.0 / (1.0 + exp(-src[i]));
    }
}

CNN_ACTIV_DEF(cnn_sigmoid_grad)
{
    int i;

    // Find sigmoid gradient
    cnn_sigmoid(buf, src, len, NULL);
    for (i = 0; i < len; i++)
    {
        dst[i] = buf[i] * (1.0 - buf[i]);
    }
}

CNN_ACTIV_DEF(cnn_tanh)
{
    int i;
    for (i = 0; i < len; i++)
    {
        dst[i] = 2.0 / (1.0 + exp(-2.0 * src[i])) - 1.0;
    }
}

CNN_ACTIV_DEF(cnn_tanh_grad)
{
    int i;

    // Find tanh gradient
    cnn_tanh(buf, src, len, NULL);
    for (i = 0; i < len; i++)
    {
        dst[i] = 1.0 - buf[i] * buf[i];
    }
}

CNN_ACTIV_DEF(cnn_gaussian)
{
    int i;
    for (i = 0; i < len; i++)
    {
        dst[i] = exp(-pow(src[i], 2.0) * 0.5);
    }
}

CNN_ACTIV_DEF(cnn_gaussian_grad)
{
    int i;

    // Find gaussian gradient
    cnn_gaussian(buf, src, len, NULL);
    for (i = 0; i < len; i++)
    {
        dst[i] = -src[i] * buf[i];
    }
}

CNN_ACTIV_DEF(cnn_bent_identity)
{
    int i;
    for (i = 0; i < len; i++)
    {
        dst[i] = (sqrt(pow(src[i], 2) + 1.0) - 1.0) / 2.0 + src[i];
    }
}

CNN_ACTIV_DEF(cnn_bent_identity_grad)
{
    int i;

    // Find bent indentity gradient
    for (i = 0; i < len; i++)
    {
        dst[i] = src[i] / (2.0 * sqrt(pow(src[i], 2.0) + 1.0)) + 1.0;
    }
}

CNN_ACTIV_DEF(cnn_softplus)
{
    int i;
    for (i = 0; i < len; i++)
    {
        dst[i] = log1p(exp(src[i]));
    }
}

CNN_ACTIV_DEF(cnn_softplus_grad)
{
    int i;

    // Find softplus gradient
    for (i = 0; i < len; i++)
    {
        dst[i] = 1.0 / (1.0 + exp(-src[i]));
    }
}

CNN_ACTIV_DEF(cnn_softsign)
{
    int i;
    for (i = 0; i < len; i++)
    {
        dst[i] = src[i] / (1.0 + fabs(src[i]));
    }
}

CNN_ACTIV_DEF(cnn_softsign_grad)
{
    int i;

    // Find softsign gradient
    for (i = 0; i < len; i++)
    {
        dst[i] = 1.0 / pow(1.0 + fabs(src[i]), 2.0);
    }
}

CNN_ACTIV_DEF(cnn_sinc)
{
    int i;
    for (i = 0; i < len; i++)
    {
        if (src[i] == 0.0)
        {
            dst[i] = 1.0;
        }
        else
        {
            dst[i] = sin(src[i]) / src[i];
        }
    }
}

CNN_ACTIV_DEF(cnn_sinc_grad)
{
    int i;

    // Find sinc gradient
    for (i = 0; i < len; i++)
    {
        if (src[i] == 0.0)
        {
            dst[i] = 0.0;
        }
        else
        {
            dst[i] = (cos(src[i]) / src[i]) - (sin(src[i]) / pow(src[i], 2.0));
        }
    }
}

CNN_ACTIV_DEF(cnn_sinusoid)
{
    int i;
    for (i = 0; i < len; i++)
    {
        dst[i] = sin(src[i]);
    }
}

CNN_ACTIV_DEF(cnn_sinusoid_grad)
{
    int i;

    // Find sinusoid gradient
    for (i = 0; i < len; i++)
    {
        dst[i] = cos(src[i]);
    }
}

int cnn_get_activ_id(const char* name)
{
    int i;
    int ret = CNN_PARSE_FAILED;

    if (name != NULL)
    {
        for (i = 0; i < CNN_ACTIV_AMOUNT; i++)
        {
            ret = strcmp(name, cnn_activ_name[i]);
            if (ret == 0)
            {
                ret = i;
                goto RET;
            }
        }
    }

RET:
    return ret;
}
