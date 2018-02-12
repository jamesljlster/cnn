#ifndef __CNN_BUILTIN_MATH_H__
#define __CNN_BUILTIN_MATH_H__

#define CNN_AFUNC_AMOUNT 3

/**
 * CNN Activation Function Define Macro
 *     dst: Matrix with size len * len
 *     src: Vector with size len
 *     len: Size of output layer
 *     buf: Provide calculation buffer.
 */
#define CNN_AFUNC_DEF(name) void name(float* dst, float* src, int len, float* buf)

extern CNN_AFUNC_DEF((*cnn_afunc_list[]));
extern CNN_AFUNC_DEF((*cnn_afunc_grad_list[]));
extern const char* cnn_afunc_name[];

#ifdef __cplusplus
extern "C" {
#endif

CNN_AFUNC_DEF(cnn_softmax);
CNN_AFUNC_DEF(cnn_softmax_grad);

CNN_AFUNC_DEF(cnn_relu);
CNN_AFUNC_DEF(cnn_relu_grad);

CNN_AFUNC_DEF(cnn_swish);
CNN_AFUNC_DEF(cnn_swish_grad);

int cnn_get_afunc_id(const char* name);

#ifdef __cplusplus
}
#endif

#endif
