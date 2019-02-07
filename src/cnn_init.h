#ifndef __CNN_INIT__
#define __CNN_INIT__

#include "cnn_types.h"

#ifdef CNN_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

struct CNN_INIT
{
    int inited;
    int randSeed;

#ifdef CNN_WITH_CUDA
    cublasHandle_t blasHandle;
#endif
};

struct CNN_BOX_MULLER
{
    int saved;
    double val;
};

extern struct CNN_INIT cnnInit;

#ifdef __cplusplus
extern "C"
{
#endif

    float cnn_normal_distribution(struct CNN_BOX_MULLER* bmPtr, double mean,
                                  double stddev);
    float cnn_xavier_init(struct CNN_BOX_MULLER* bmPtr, int inSize,
                          int outSize);
    float cnn_zero(void);

#ifdef __cplusplus
}
#endif

#endif
