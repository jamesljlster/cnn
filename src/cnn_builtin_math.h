#ifndef __CNN_BUILTIN_MATH_H__
#define __CNN_BUILTIN_MATH_H__

#define CNN_TFUNC_AMOUNT 3

#define CNN_AFUNC_DEF(name) void name(float* dst, float* src, int len)

extern CNN_AFUNC_DEF((*cnn_transfer_list[]));
extern CNN_AFUNC_DEF((*cnn_transfer_derivative_list[]));
extern const char* cnn_transfer_func_name[];

#ifdef __cplusplus
extern "C" {
#endif

CNN_AFUNC_DEF(cnn_softmax);
CNN_AFUNC_DEF(cnn_softmax_derivative);

CNN_AFUNC_DEF(cnn_relu);
CNN_AFUNC_DEF(cnn_relu_derivative);

CNN_AFUNC_DEF(cnn_swish);
CNN_AFUNC_DEF(cnn_swish_derivative);

#ifdef __cplusplus
}
#endif

#endif
