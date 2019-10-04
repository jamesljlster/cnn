#include <math.h>
#include <string.h>

#include <cblas.h>

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
    cnn_sinusoid,       //
    cnn_identity        //
};

CNN_ACTIV_GRAD_DEF((*cnn_activ_grad_list[])) = {
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
    cnn_sinusoid_grad,       //
    cnn_identity_grad        //
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
    "Sinusoid",            //
    "Identity"             //
};

CNN_ACTIV_DEF(cnn_softmax)
{
    int i;
    float max, sum;

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
    //#endif
}

CNN_ACTIV_GRAD_DEF(cnn_softmax_grad)
{
    int i, j;

    // Find softmax gradient matrix
    for (i = 0; i < len; i++)
    {
        for (j = 0; j < len; j++)
        {
            buf[i * len + j] = cache[i] * ((float)(i == j) - cache[j]);
        }
    }

    // Find layer gradient
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,  //
                1, len, len,                                //
                1.0,                                        //
                gradIn, len,                                //
                buf, len,                                   //
                0.0,                                        //
                gradOut, len);
}

CNN_ACTIV_DEF(cnn_relu)
{
#ifdef CNN_WITH_CUDA
    cnn_relu_gpu(dst, src, len);
#else
#pragma omp parallel for shared(dst, src)
    for (int i = 0; i < len; i++)
    {
        dst[i] = fmaxf(src[i], 0.0f);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_relu_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_relu_grad_gpu(dst, src, buf, len);
#else
    // Find relu gradient
#pragma omp parallel for shared(gradOut, gradIn, src)
    for (int i = 0; i < len; i++)
    {
        gradOut[i] = ((src[i] < 0.0f) ? 0 : 1) * gradIn[i];
    }
#endif
}

CNN_ACTIV_DEF(cnn_swish)
{
#ifdef CNN_WITH_CUDA
    cnn_swish_gpu(dst, src, len);
#else
    float srcVal;

#pragma omp parallel for shared(dst, src) private(srcVal)
    for (int i = 0; i < len; i++)
    {
        srcVal = src[i];
        dst[i] = srcVal / (1.0f + expf(-srcVal));
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_swish_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_swish_grad_gpu(dst, src, buf, len);
#else
    // Find swish gradient
    float srcVal, cacheVal;

#pragma omp parallel for shared(gradOut, gradIn, src, cache) private(srcVal, \
                                                                     cacheVal)
    for (int i = 0; i < len; i++)
    {
        srcVal = src[i];
        if (srcVal == 0.0f)
        {
            gradOut[i] = 0.5 * gradIn[i];
        }
        else
        {
            cacheVal = cache[i];
            gradOut[i] = (cacheVal + (cacheVal / srcVal) * (1.0f - cacheVal)) *
                         gradIn[i];
        }
    }
#endif
}

CNN_ACTIV_DEF(cnn_sigmoid)
{
#ifdef CNN_WITH_CUDA
    cnn_sigmoid_gpu(dst, src, len);
#else
#pragma omp parallel for shared(dst, src)
    for (int i = 0; i < len; i++)
    {
        dst[i] = 1.0 / (1.0 + exp(-src[i]));
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_sigmoid_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_sigmoid_grad_gpu(dst, src, buf, len);
#else
    float cacheVal;

    // Find sigmoid gradient
#pragma omp parallel for shared(gradOut, gradIn, cache) private(cacheVal)
    for (int i = 0; i < len; i++)
    {
        cacheVal = cache[i];
        gradOut[i] = (cacheVal * (1.0 - cacheVal)) * gradIn[i];
    }
#endif
}

CNN_ACTIV_DEF(cnn_tanh)
{
#ifdef CNN_WITH_CUDA
    cnn_tanh_gpu(dst, src, len);
#else
#pragma omp parallel for shared(dst, src)
    for (int i = 0; i < len; i++)
    {
        dst[i] = 2.0 / (1.0 + exp(-2.0 * src[i])) - 1.0;
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_tanh_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_tanh_grad_gpu(dst, src, buf, len);
#else
    float cacheVal;

    // Find tanh gradient
#pragma omp parallel for shared(gradOut, gradIn, cache) private(cacheVal)
    for (int i = 0; i < len; i++)
    {
        cacheVal = buf[i];
        gradOut[i] = (1.0 - cacheVal * cacheVal) * gradIn[i];
    }
#endif
}

CNN_ACTIV_DEF(cnn_gaussian)
{
#ifdef CNN_WITH_CUDA
    cnn_gaussian_gpu(dst, src, len);
#else
#pragma omp parallel for shared(dst, src)
    for (int i = 0; i < len; i++)
    {
        dst[i] = exp(-pow(src[i], 2.0) * 0.5);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_gaussian_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_gaussian_grad_gpu(dst, src, buf, len);
#else
    // Find gaussian gradient
#pragma omp parallel for shared(gradOut, gradIn, src, cache)
    for (int i = 0; i < len; i++)
    {
        gradOut[i] = (-src[i] * cache[i]) * gradIn[i];
    }
#endif
}

CNN_ACTIV_DEF(cnn_bent_identity)
{
#ifdef CNN_WITH_CUDA
    cnn_bent_identity_gpu(dst, src, len);
#else
    float srcVal;

#pragma omp parallel for shared(dst, src) private(srcVal)
    for (int i = 0; i < len; i++)
    {
        srcVal = src[i];
        dst[i] = (sqrt(pow(srcVal, 2) + 1.0) - 1.0) / 2.0 + srcVal;
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_bent_identity_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_bent_identity_grad_gpu(dst, src, buf, len);
#else
    // Find bent indentity gradient
    float srcVal;

#pragma omp parallel for shared(gradOut, gradIn, src) private(srcVal)
    for (int i = 0; i < len; i++)
    {
        srcVal = src[i];
        gradOut[i] =
            (srcVal / (2.0 * sqrt(pow(srcVal, 2.0) + 1.0)) + 1.0) * gradIn[i];
    }
#endif
}

CNN_ACTIV_DEF(cnn_softplus)
{
#ifdef CNN_WITH_CUDA
    cnn_softplus_gpu(dst, src, len);
#else
#pragma omp parallel for shared(dst, src)
    for (int i = 0; i < len; i++)
    {
        dst[i] = log1p(exp(src[i]));
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_softplus_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_softplus_grad_gpu(dst, src, buf, len);
#else
    // Find softplus gradient
#pragma omp parallel for shared(gradOut, gradIn, src)
    for (int i = 0; i < len; i++)
    {
        gradOut[i] = (1.0 / (1.0 + exp(-src[i]))) * gradIn[i];
    }
#endif
}

CNN_ACTIV_DEF(cnn_softsign)
{
#ifdef CNN_WITH_CUDA
    cnn_softsign_gpu(dst, src, len);
#else
    float srcVal;

#pragma omp parallel for shared(dst, src) private(srcVal)
    for (int i = 0; i < len; i++)
    {
        srcVal = src[i];
        dst[i] = srcVal / (1.0 + fabs(srcVal));
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_softsign_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_softsign_grad_gpu(dst, src, buf, len);
#else
    // Find softsign gradient
#pragma omp parallel for shared(gradOut, gradIn, src)
    for (int i = 0; i < len; i++)
    {
        gradOut[i] = (1.0 / pow(1.0 + fabs(src[i]), 2.0)) * gradIn[i];
    }
#endif
}

CNN_ACTIV_DEF(cnn_sinc)
{
#ifdef CNN_WITH_CUDA
    cnn_sinc_gpu(dst, src, len);
#else
    float srcVal;

#pragma omp parallel for shared(src, dst) private(srcVal)
    for (int i = 0; i < len; i++)
    {
        srcVal = src[i];
        if (srcVal == 0.0)
        {
            dst[i] = 1.0;
        }
        else
        {
            dst[i] = sin(srcVal) / srcVal;
        }
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_sinc_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_sinc_grad_gpu(dst, src, buf, len);
#else
    // Find sinc gradient
    float srcVal;

#pragma omp parallel for shared(gradOut, gradIn, src) private(srcVal)
    for (int i = 0; i < len; i++)
    {
        srcVal = src[i];
        if (srcVal == 0.0)
        {
            gradOut[i] = 0.0;
        }
        else
        {
            gradOut[i] =
                ((cos(srcVal) / srcVal) - (sin(srcVal) / pow(srcVal, 2.0))) *
                gradIn[i];
        }
    }
#endif
}

CNN_ACTIV_DEF(cnn_sinusoid)
{
#ifdef CNN_WITH_CUDA
    cnn_sin_gpu(dst, src, len);
#else
#pragma omp parallel for shared(dst, src)
    for (int i = 0; i < len; i++)
    {
        dst[i] = sin(src[i]);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_sinusoid_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_sin_grad_gpu(dst, src, buf, len);
#else
    // Find sinusoid gradient
#pragma omp parallel for shared(gradOut, gradIn, src)
    for (int i = 0; i < len; i++)
    {
        gradOut[i] = cos(src[i]) * gradIn[i];
    }
#endif
}

CNN_ACTIV_DEF(cnn_identity)
{
#ifdef CNN_WITH_CUDA
    cnn_identity_gpu(dst, src, len);
#else
#pragma omp parallel for shared(dst, src)
    for (int i = 0; i < len; i++)
    {
        dst[i] = src[i];
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_identity_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_identity_grad_gpu(dst, src, buf, len);
#else
    // Find identity gradient
#pragma omp parallel for shared(gradOut, gradIn)
    for (int i = 0; i < len; i++)
    {
        gradOut[i] = gradIn[i];
    }
#endif
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
