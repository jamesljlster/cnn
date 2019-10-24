#ifndef __CNN_RBFACT_H__
#define __CNN_RBFACT_H__

#include <math.h>
#include <string.h>

#include "cnn_types.h"

static inline void cnn_rbfact_forward_inference_cpu(  //
    float* dst, int dstChannel, float* src, int srcChannel, float* center,
    float* var, int batch, int height, int width)
{
    for (int i = 0; i < batch * dstChannel * height * width; i++)
    {
        // Indexing
        int e = i % (height * width);
        int c = (i / (height * width)) % dstChannel;
        int n = i / (dstChannel * height * width);

        // RBF Calculation
        float pwrDist = 0;
        for (int inC = 0; inC < srcChannel; inC++)
        {
            float srcVal = src[n * (srcChannel * height * width) +  //
                               inC * (height * width) +             //
                               e];

            pwrDist += (srcVal - center[c * srcChannel + inC]) *
                       (srcVal - center[c * srcChannel + inC]);
        }

        dst[n * (dstChannel * height * width) +  //
            c * (height * width) +               //
            e] = exp(-1.0 * pwrDist / (2.0 * var[c]));
    }
}

static inline void cnn_rbfact_forward_training_cpu(  //
    float* dst, int dstChannel, float* src, int srcChannel, float* center,
    float* runVar, float* saveVar, float* varBuf, int batch, int height,
    int width, float expAvgFactor)
{
    // Clear buffer
    memset(varBuf, 0, sizeof(float) * dstChannel);

    // Copy variance
    memcpy(saveVar, runVar, sizeof(float) * dstChannel);

    // RBFAct forward
    for (int i = 0; i < batch * dstChannel * height * width; i++)
    {
        // Indexing
        int e = i % (height * width);
        int c = (i / (height * width)) % dstChannel;
        int n = i / (dstChannel * height * width);

        // RBF Calculation
        float pwrDist = 0;
        for (int inC = 0; inC < srcChannel; inC++)
        {
            float srcVal = src[n * (srcChannel * height * width) +  //
                               inC * (height * width) +             //
                               e];

            pwrDist += (srcVal - center[c * srcChannel + inC]) *
                       (srcVal - center[c * srcChannel + inC]);

            varBuf[c] += pwrDist;
        }

        dst[n * (dstChannel * height * width) +  //
            c * (height * width) +               //
            e] = exp(-1.0 * pwrDist / (2.0 * runVar[c]));
    }

    // Find new running variance
    for (int c = 0; c < dstChannel; c++)
    {
        float tmpVar = varBuf[c] / (float)(batch * height * width);
        runVar[c] = runVar[c] * (1.0 - expAvgFactor) + tmpVar * expAvgFactor;
    }
}

static inline void cnn_rbfact_backward_layer_cpu(  //
    float* gradOut, int gradOutCh, float* gradIn, int gradInCh, float* src,
    float* cache, float* center, float* saveVar, int batch, int height,
    int width)
{
    // RBFAct layer gradient
    for (int i = 0; i < batch * gradOutCh * height * width; i++)
    {
        // Indexing
        int e = i % (height * width);
        int c = (i / (height * width)) % gradOutCh;
        int n = i / (gradOutCh * height * width);

        // RBF gradient calculation
        float gradSum = 0;
        float srcVal = src[n * (gradOutCh * height * width) +  //
                           c * (height * width) +              //
                           e];

        for (int gInC = 0; gInC < gradInCh; gInC++)
        {
            float cacheVal = cache[n * (gradInCh * height * width) +  //
                                   gInC * (height * width) +          //
                                   e];
            float gInVal = gradIn[n * (gradInCh * height * width) +  //
                                  gInC * (height * width) +          //
                                  e];
            float ctr = center[gInC * gradOutCh + c];

            gradSum += gInVal * cacheVal * ((srcVal - ctr) / saveVar[gInC]) *
                       (((srcVal - ctr) * (srcVal - ctr) /
                         ((float)(batch * height * width) * saveVar[gInC])) -
                        1);
        }

        gradOut[n * (gradOutCh * height * width) +  //
                c * (height * width) +              //
                e] = gradSum;
    }
}

static inline void cnn_rbfact_backward_center_cpu(  //
    float* center, float* centerGrad, float* workSpace, int srcChannel,
    int dstChannel, float* src, int batch, int height, int width)
{
    // Cache
    float* ctrSquareSum = workSpace;

    // Find squared sum of centers
    for (int i = 0; i < dstChannel; i++)
    {
        float sum = 0;
        for (int j = 0; j < srcChannel; j++)
        {
            sum += center[i * srcChannel + j] * center[i * srcChannel + j];
        }

        ctrSquareSum[i] = sum;
    }

    // Do clustering
    for (int i = 0; i < batch * dstChannel * height * width; i++)
    {
        float sim = 0;

        // Indexing
        int e = i % (height * width);
        int c = (i / (height * width)) % dstChannel;
        int n = i / (dstChannel * height * width);

        // Find vector similarity
        float srcSquaredSum = 0;
        for (int srcCh = 0; srcCh < srcChannel; srcCh++)
        {
            int srcIndex = n * (srcChannel * height * width) +  //
                           srcCh * (height * width) +           //
                           e;

            srcSquaredSum += src[srcIndex] * src[srcIndex];
            sim += src[srcIndex] * center[c * srcChannel + srcCh];
        }

        sim /= ctrSquareSum[c] * srcSquaredSum;

        // Sum difference
        for (int srcCh = 0; srcCh < srcChannel; srcCh++)
        {
            int srcIndex = n * (srcChannel * height * width) +  //
                           srcCh * (height * width) +           //
                           e;

            centerGrad[c * srcChannel + srcCh] +=
                sim * (src[srcIndex] - center[c * srcChannel + srcCh]);
        }
    }
}

#endif
