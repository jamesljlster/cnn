#ifndef __CNN_BUILTIN_MATH_INLINE_H__
#define __CNN_BUILTIN_MATH_INLINE_H__

#include <math.h>

#include "cnn_config.h"

#ifdef CNN_WITH_CUDA
#include <cuda_runtime.h>
#define __CNN_MATH_API __device__ void
#else
#define __CNN_MATH_API static inline void
#endif

#define CNN_ACTIV_INLINE_DEF(name) \
    __CNN_MATH_API __cnn_##name(float* dst, float* src)
#define CNN_ACTIV_GRAD_INLINE_DEF(name)                               \
    __CNN_MATH_API __cnn_##name##_grad(float* gradOut, float* gradIn, \
                                       float* src, float* cache)

// Relu
CNN_ACTIV_INLINE_DEF(relu) { *dst = fmaxf(*src, 0.0f); }
CNN_ACTIV_GRAD_INLINE_DEF(relu)
{
    if (*src < 0.0f)
    {
        *gradOut = 0;
    }
    else
    {
        *gradOut = *gradIn;
    }
}

// Swish
CNN_ACTIV_INLINE_DEF(swish) { *dst = *src / (1.0f + expf(-(*src))); }
CNN_ACTIV_GRAD_INLINE_DEF(swish)
{
    if (*src == 0.0f)
    {
        *gradOut = 0.5 * *gradIn;
    }
    else
    {
        *gradOut = (*cache + (*cache / *src) * (1.0f - *cache)) * *gradIn;
    }
}

// Sigmoid
CNN_ACTIV_INLINE_DEF(sigmoid) { *dst = 1.0 / (1.0 + exp(-*src)); }
CNN_ACTIV_GRAD_INLINE_DEF(sigmoid)
{
    *gradOut = (*cache * (1.0 - *cache)) * *gradIn;
}

// Hyperbolic tangent
CNN_ACTIV_INLINE_DEF(tanh) { *dst = 2.0 / (1.0 + exp(-2.0 * *src)) - 1.0; }
CNN_ACTIV_GRAD_INLINE_DEF(tanh)
{
    *gradOut = (1.0 - *cache * *cache) * *gradIn;
}

// Gaussian
CNN_ACTIV_INLINE_DEF(gaussian) { *dst = exp(-*src * *src * 0.5); }
CNN_ACTIV_GRAD_INLINE_DEF(gaussian) { *gradOut = (-*src * *cache) * *gradIn; }

// Bent identity
CNN_ACTIV_INLINE_DEF(bent_identity)
{
    *dst = (sqrt(*src * *src + 1.0) - 1.0) / 2.0 + *src;
}

CNN_ACTIV_GRAD_INLINE_DEF(bent_identity)
{
    *gradOut = (*src / (2.0 * sqrt(*src * *src + 1.0)) + 1.0) * *gradIn;
}

// Softplus
CNN_ACTIV_INLINE_DEF(softplus) { *dst = log1p(exp(*src)); }
CNN_ACTIV_GRAD_INLINE_DEF(softplus)
{
    *gradOut = (1.0 / (1.0 + exp(-*src))) * *gradIn;
}

// Softsign
CNN_ACTIV_INLINE_DEF(softsign) { *dst = *src / (1.0 + fabs(*src)); }
CNN_ACTIV_GRAD_INLINE_DEF(softsign)
{
    *gradOut = (1.0 / pow(1.0 + fabs(*src), 2.0)) * *gradIn;
}

// Sinc
CNN_ACTIV_INLINE_DEF(sinc)
{
    if (*src == 0.0)
    {
        *dst = 1.0;
    }
    else
    {
        *dst = sin(*src) / *src;
    }
}

CNN_ACTIV_GRAD_INLINE_DEF(sinc)
{
    if (*src == 0.0)
    {
        *gradOut = 0.0;
    }
    else
    {
        *gradOut =
            ((cos(*src) / *src) - (sin(*src) / pow(*src, 2.0))) * *gradIn;
    }
}

// Sinusoid
CNN_ACTIV_INLINE_DEF(sinusoid) { *dst = sin(*src); }
CNN_ACTIV_GRAD_INLINE_DEF(sinusoid) { *gradOut = cos(*src) * *gradIn; }

// Identity
CNN_ACTIV_INLINE_DEF(identity) { *dst = *src; }
CNN_ACTIV_GRAD_INLINE_DEF(identity) { *gradOut = *gradIn; }

#endif
