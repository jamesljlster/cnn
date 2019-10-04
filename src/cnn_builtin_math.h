#ifndef __CNN_BUILTIN_MATH_H__
#define __CNN_BUILTIN_MATH_H__

#define CNN_ACTIV_AMOUNT 12

/**
 * CNN Activation Function Define Macro
 *  - Forward
 *     dst: Vector with size len
 *     src: Vector with size len
 *     len: Size of output layer
 *     buf: Provide calculation buffer.
 */
#define CNN_ACTIV_DEF(name) \
    void name(float* dst, float* src, int len, float* buf)
#define CNN_ACTIV_GRAD_DEF(name)                                  \
    void name(float* gradOut, float* gradIn, float* src, int len, \
              float* cache, float* buf)
#define CNN_ACTIV_GROUP_DEF(name) \
    CNN_ACTIV_DEF(name);          \
    CNN_ACTIV_GRAD_DEF(name##_grad);

extern CNN_ACTIV_DEF((*cnn_activ_list[]));
extern CNN_ACTIV_GRAD_DEF((*cnn_activ_grad_list[]));

extern const char* cnn_activ_name[];

#ifdef __cplusplus
extern "C"
{
#endif

    CNN_ACTIV_GROUP_DEF(cnn_softmax);
    CNN_ACTIV_GROUP_DEF(cnn_relu);
    CNN_ACTIV_GROUP_DEF(cnn_swish);
    CNN_ACTIV_GROUP_DEF(cnn_sigmoid);
    CNN_ACTIV_GROUP_DEF(cnn_tanh);
    CNN_ACTIV_GROUP_DEF(cnn_gaussian);
    CNN_ACTIV_GROUP_DEF(cnn_bent_identity);
    CNN_ACTIV_GROUP_DEF(cnn_softplus);
    CNN_ACTIV_GROUP_DEF(cnn_softsign);
    CNN_ACTIV_GROUP_DEF(cnn_sinc);
    CNN_ACTIV_GROUP_DEF(cnn_sinusoid);
    CNN_ACTIV_GROUP_DEF(cnn_identity);

    int cnn_get_activ_id(const char* name);

#ifdef __cplusplus
}
#endif

#endif
