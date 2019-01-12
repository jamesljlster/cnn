#ifndef __CNN_INIT__
#define __CNN_INIT__

#include "cnn_types.h"

struct CNN_BOX_MULLER
{
    int saved;
    double val;
};

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
