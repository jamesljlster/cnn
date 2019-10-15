#ifndef __CNN_RBFACT_H__
#define __CNN_RBFACT_H__

#include <math.h>

#include "cnn_types.h"

static inline void cnn_forward_rbfact_cpu(float* dst, int dstChannel,
                                          float* src, int srcChannel,
                                          float* center, float* sigma,
                                          int batch, int height, int width)
{
    for (int i = 0; i < batch * dstChannel * height * width; i++)
    {
        // Indexing
        int w = i % width;
        int h = (i / width) % height;
        int c = (i / (height * width)) % dstChannel;
        int n = i / (dstChannel * height * width);

        // RBF Calculation
        float pwrDist = 0;
        for (int inC = 0; inC < srcChannel; inC++)
        {
            float srcVal = src[n * (srcChannel * height * width) +  //
                               inC * (height * width) +             //
                               h * width +                          //
                               w];

            pwrDist += (srcVal - center[c * srcChannel + inC]) *
                       (srcVal - center[c * srcChannel + inC]);
        }

        dst[n * (dstChannel * height * width) +  //
            c * (height * width) +               //
            h * width +                          //
            w] = exp(-1.0 * pwrDist / (2.0 * sigma[c]));
    }
}

#endif
