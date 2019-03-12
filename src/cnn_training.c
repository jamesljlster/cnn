#include <stdlib.h>
#include <string.h>

#include "cnn.h"
#include "cnn_private.h"

int cnn_training(cnn_t cnn, float lRate, float* inputMat, float* desireMat,
                 float* outputMat, float* errMat, float gradLimit)
{
    int i;
    int outSize;
    int ret = CNN_NO_ERROR;

    float* outStore = NULL;
    float* errStore = NULL;

    float* outMem = NULL;
    float* errMem = NULL;

    struct CNN_CONFIG* cfgRef;

    // Get reference
    cfgRef = &cnn->cfg;
    outSize = cnn->layerList[cfgRef->layers - 1].outMat.data.rows *
              cnn->layerList[cfgRef->layers - 1].outMat.data.cols;

    // Memory allocation
    if (outputMat != NULL)
    {
        outStore = outputMat;
    }
    else
    {
        cnn_alloc(outMem, outSize, float, ret, RET);
        outStore = outMem;
    }

    if (errMat != NULL)
    {
        errStore = errMat;
    }
    else
    {
        cnn_alloc(errMem, outSize, float, ret, RET);
        errStore = errMem;
    }

    // Forward
    cnn_forward(cnn, inputMat, outStore);

    // Find error
    for (i = 0; i < outSize; i++)
    {
        errStore[i] = desireMat[i] - outStore[i];
    }

    // Backpropagation
    cnn_backward(cnn, errStore);

    // Update network
    cnn_update(cnn, lRate, gradLimit);

RET:
    cnn_free(outMem);
    cnn_free(errMem);

    return ret;
}
