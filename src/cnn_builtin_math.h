#ifndef __CNN_BUILTIN_MATH_H__
#define __CNN_BUILTIN_MATH_H__

#define CNN_AFUNC_DEF(name) void name(float* dst, float* src, int len)

#ifdef __cplusplus
extern "C" {
#endif

CNN_AFUNC_DEF(cnn_softmax);

CNN_AFUNC_DEF(cnn_relu);

#ifdef __cplusplus
}
#endif

#endif
