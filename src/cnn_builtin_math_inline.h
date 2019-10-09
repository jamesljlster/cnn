#ifndef __CNN_BUILTIN_MATH_INLINE_H__
#define __CNN_BUILTIN_MATH_INLINE_H__

#include <math.h>

#define CNN_ACTIV_INLINE_DEF(name) \
    static inline void __cnn_##name(float* dst, float* src)
#define CNN_ACTIV_GRAD_INLINE_DEF(name)                                   \
    static inline void __cnn_##name##_grad(float* gradOut, float* gradIn, \
                                           float* src, float* cache)

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

CNN_ACTIV_INLINE_DEF(sigmoid) { *dst = 1.0 / (1.0 + exp(-*src)); }

#endif
