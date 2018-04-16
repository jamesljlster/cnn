#ifndef __CNN_BUILTIN_MATH_H__
#define __CNN_BUILTIN_MATH_H__

#define CNN_ACTIV_AMOUNT 3

/**
 * CNN Activation Function Define Macro
 *     dst: Matrix with size len * len
 *     src: Vector with size len
 *     len: Size of output layer
 *     buf: Provide calculation buffer.
 */
#define CNN_ACTIV_DEF(name) void name(float* dst, float* src, int len, float* buf)

extern CNN_ACTIV_DEF((*cnn_activ_list[]));
extern CNN_ACTIV_DEF((*cnn_activ_grad_list[]));
extern const char* cnn_activ_name[];

#ifdef __cplusplus
extern "C" {
#endif

CNN_ACTIV_DEF(cnn_softmax);
CNN_ACTIV_DEF(cnn_softmax_grad);

CNN_ACTIV_DEF(cnn_relu);
CNN_ACTIV_DEF(cnn_relu_grad);

CNN_ACTIV_DEF(cnn_swish);
CNN_ACTIV_DEF(cnn_swish_grad);

CNN_ACTIV_DEF(cnn_sigmoid);
CNN_ACTIV_DEF(cnn_sigmoid_grad);

CNN_ACTIV_DEF(cnn_tanh);
CNN_ACTIV_DEF(cnn_tanh_grad);

CNN_ACTIV_DEF(cnn_gaussian);
CNN_ACTIV_DEF(cnn_gaussian_grad);

CNN_ACTIV_DEF(cnn_bent_identity);
CNN_ACTIV_DEF(cnn_bent_identity_grad);

CNN_ACTIV_DEF(cnn_softplus);
CNN_ACTIV_DEF(cnn_softplus_grad);

CNN_ACTIV_DEF(cnn_softsign);
CNN_ACTIV_DEF(cnn_softsign_grad);

CNN_ACTIV_DEF(cnn_sinc);
CNN_ACTIV_DEF(cnn_sinc_grad);

CNN_ACTIV_DEF(cnn_sinusoid);
CNN_ACTIV_DEF(cnn_sinusoid_grad);

int cnn_get_activ_id(const char* name);

#ifdef __cplusplus
}
#endif

#endif
