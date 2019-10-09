#include <math.h>
#include <string.h>

#include <cblas.h>

#include "cnn.h"
#include "cnn_builtin_math.h"
#include "cnn_builtin_math_inline.h"
#include "cnn_config.h"
#include "cnn_init.h"
#include "cnn_macro.h"

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

CNN_ACTIV_GRAD_DEF(cnn_softmax_grad)
{
#ifdef CNN_WITH_CUDA
    float alpha = 1.0;
    float beta = 0.0;

    // Find derivative matrix
    cnn_smax_grad_gpu(buf, cache, len);

    // Find layer gradient
    cnn_assert_cu(cublasSgemm(cnnInit.blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,  //
                              len, 1, len,                                   //
                              &alpha,                                        //
                              buf, len,                                      //
                              gradIn, len,                                   //
                              &beta,                                         //
                              gradOut, len));

#else
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
#endif
}

CNN_ACTIV_DEF(cnn_relu)
{
#ifdef CNN_WITH_CUDA
    cnn_relu_gpu(dst, src, len);
#else
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_relu(dst + i, src + i);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_relu_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_relu_grad_gpu(gradOut, gradIn, src, len, cache);
#else
    // Find relu gradient
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_relu_grad(gradOut + i, gradIn + i, src + i, NULL);
    }
#endif
}

CNN_ACTIV_DEF(cnn_swish)
{
#ifdef CNN_WITH_CUDA
    cnn_swish_gpu(dst, src, len);
#else
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_swish(dst + i, src + i);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_swish_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_swish_grad_gpu(gradOut, gradIn, src, len, cache);
#else
    // Find swish gradient
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_swish_grad(gradOut + i, gradIn + i, src + i, cache + i);
    }
#endif
}

CNN_ACTIV_DEF(cnn_sigmoid)
{
#ifdef CNN_WITH_CUDA
    cnn_sigmoid_gpu(dst, src, len);
#else
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_sigmoid(dst + i, src + i);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_sigmoid_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_sigmoid_grad_gpu(gradOut, gradIn, src, len, cache);
#else
    // Find sigmoid gradient
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_sigmoid_grad(gradOut + i, gradIn + i, NULL, cache + i);
    }
#endif
}

CNN_ACTIV_DEF(cnn_tanh)
{
#ifdef CNN_WITH_CUDA
    cnn_tanh_gpu(dst, src, len);
#else
#pragma omp parallel
    for (int i = 0; i < len; i++)
    {
        __cnn_tanh(dst + i, src + i);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_tanh_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_tanh_grad_gpu(gradOut, gradIn, src, len, cache);
#else
    // Find tanh gradient
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_tanh_grad(gradOut + i, gradIn + i, NULL, cache + i);
    }
#endif
}

CNN_ACTIV_DEF(cnn_gaussian)
{
#ifdef CNN_WITH_CUDA
    cnn_gaussian_gpu(dst, src, len);
#else
#pragma omp parallel
    for (int i = 0; i < len; i++)
    {
        __cnn_gaussian(dst + i, src + i);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_gaussian_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_gaussian_grad_gpu(gradOut, gradIn, src, len, cache);
#else
    // Find gaussian gradient
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_gaussian_grad(gradOut + i, gradIn + i, src + i, cache + i);
    }
#endif
}

CNN_ACTIV_DEF(cnn_bent_identity)
{
#ifdef CNN_WITH_CUDA
    cnn_bent_identity_gpu(dst, src, len);
#else
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_bent_identity(dst + i, src + i);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_bent_identity_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_bent_identity_grad_gpu(gradOut, gradIn, src, len, cache);
#else
    // Find bent indentity gradient
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_bent_identity_grad(gradOut + i, gradIn + i, src + i, NULL);
    }
#endif
}

CNN_ACTIV_DEF(cnn_softplus)
{
#ifdef CNN_WITH_CUDA
    cnn_softplus_gpu(dst, src, len);
#else
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_softplus(dst + i, src + i);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_softplus_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_softplus_grad_gpu(gradOut, gradIn, src, len, cache);
#else
    // Find softplus gradient
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_softplus_grad(gradOut + i, gradIn + i, src + i, NULL);
    }
#endif
}

CNN_ACTIV_DEF(cnn_softsign)
{
#ifdef CNN_WITH_CUDA
    cnn_softsign_gpu(dst, src, len);
#else
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_softsign(dst + i, src + i);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_softsign_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_softsign_grad_gpu(gradOut, gradIn, src, len, cache);
#else
    // Find softsign gradient
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_softsign_grad(gradOut + i, gradIn + i, src + i, NULL);
    }
#endif
}

CNN_ACTIV_DEF(cnn_sinc)
{
#ifdef CNN_WITH_CUDA
    cnn_sinc_gpu(dst, src, len);
#else
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_sinc(dst + i, src + i);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_sinc_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_sinc_grad_gpu(gradOut, gradIn, src, len, cache);
#else
    // Find sinc gradient
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_sinc_grad(gradOut + i, gradIn + i, src + i, NULL);
    }
#endif
}

CNN_ACTIV_DEF(cnn_sinusoid)
{
#ifdef CNN_WITH_CUDA
    cnn_sin_gpu(dst, src, len);
#else
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_sinusoid(dst + i, src + i);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_sinusoid_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_sin_grad_gpu(gradOut, gradIn, src, len, cache);
#else
    // Find sinusoid gradient
#pragma omp parallel for shared(gradOut, gradIn, src)
    for (int i = 0; i < len; i++)
    {
        __cnn_sinusoid_grad(gradOut + i, gradIn + i, src + i, NULL);
    }
#endif
}

CNN_ACTIV_DEF(cnn_identity)
{
#ifdef CNN_WITH_CUDA
    cnn_identity_gpu(dst, src, len);
#else
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_identity(dst + i, src + i);
    }
#endif
}

CNN_ACTIV_GRAD_DEF(cnn_identity_grad)
{
#ifdef CNN_WITH_CUDA
    cnn_identity_grad_gpu(gradOut, gradIn, src, len, cache);
#else
    // Find identity gradient
#pragma omp parallel for
    for (int i = 0; i < len; i++)
    {
        __cnn_identity_grad(gradOut + i, gradIn + i, NULL, NULL);
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
