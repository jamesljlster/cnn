#ifndef __CNN_TEXT_H__
#define __CNN_TEXT_H__

#include "cnn_builtin_math.h"
#include "cnn_types.h"

static inline int cnn_text_get_index(int wShift, int hShift, int w, int h,
                                     int ch, int width, int height, int channel)
{
    int imSize;
    int row, col;

    // Get size
    imSize = width * height;

    // Find row, col index
    row = h + hShift;
    if (row < 0)
    {
        row = 0;
    }
    if (row >= height)
    {
        row = height - 1;
    }

    col = w + wShift;
    if (col < 0)
    {
        col = 0;
    }
    if (col >= width)
    {
        col = width - 1;
    }

    // Get index
    return ch * imSize + row * width + col;
}

static inline void cnn_text_unroll(int* nbrMap, int* ctrMap, int width,
                                   int height, int channel)
{
    const int wSize = 8;
    int mapCols = channel * wSize;

    for (int h = 0; h < height; h++)
    {
        int rowBase = h * width;

        for (int w = 0; w < width; w++)
        {
            int rowShift = rowBase + w;
            int nbrMemBase = rowShift * mapCols;
            int ctrMemBase = rowShift * channel;

            for (int ch = 0; ch < channel; ch++)
            {
                int nbrMemShift = nbrMemBase + ch * wSize;
                int ctrMemShift = ctrMemBase + ch;

#define __get_index(wShift, hShift) \
    cnn_text_get_index(wShift, hShift, w, h, ch, width, height, channel)

                ctrMap[ctrMemShift] = __get_index(0, 0);

                nbrMap[nbrMemShift++] = __get_index(-1, -1);
                nbrMap[nbrMemShift++] = __get_index(0, -1);
                nbrMap[nbrMemShift++] = __get_index(+1, -1);

                nbrMap[nbrMemShift++] = __get_index(-1, 0);
                nbrMap[nbrMemShift++] = __get_index(+1, 0);

                nbrMap[nbrMemShift++] = __get_index(-1, +1);
                nbrMap[nbrMemShift++] = __get_index(0, +1);
                nbrMap[nbrMemShift++] = __get_index(+1, +1);
            }
        }
    }
}

static inline void cnn_forward_text(union CNN_LAYER* layerRef,
                                    struct CNN_CONFIG* cfgRef, int layerIndex)
{
    // Cache
    const int wSize = 8;
    int activId = cfgRef->layerCfg[layerIndex].text.activId;

    int chIn = layerRef[layerIndex].text.inChannel;
    int chOut = cfgRef->layerCfg[layerIndex].text.filter;

    int wCols = wSize * chIn;

    int srcSize = layerRef[layerIndex - 1].outMat.data.cols;
    int dstSize = layerRef[layerIndex].outMat.data.cols;
    int dstImSize =
        layerRef[layerIndex].outMat.width * layerRef[layerIndex].outMat.height;

    int nbrRows = layerRef[layerIndex].outMat.width *
                  layerRef[layerIndex].outMat.height * chIn;
    int nbrCols = wSize;
    int nbrSize = nbrRows * nbrCols;

    int ctrRows = nbrRows;
    int ctrCols = 1;
    int ctrSize = ctrRows * ctrCols;

    int diffRows =
        layerRef[layerIndex].outMat.width * layerRef[layerIndex].outMat.height;
    int diffCols = chIn * wSize;
    int diffSize = diffRows * diffCols;

    int activRows =
        layerRef[layerIndex].outMat.width * layerRef[layerIndex].outMat.height;
    int activCols = chIn * wSize;
    int activSize = activRows * activCols;

    int* nbrMap = layerRef[layerIndex].text.nbrMap;
    int* ctrMap = layerRef[layerIndex].text.ctrMap;
    float* weight = layerRef[layerIndex].text.weight.mat;
    float* bias = layerRef[layerIndex].text.bias.mat;

    for (int j = 0; j < cfgRef->batch; j++)
    {
        int srcShift = j * srcSize;
        int dstShift = j * dstSize;
        int nbrShift = j * nbrSize;
        int ctrShift = j * ctrSize;
        int diffShift = j * diffSize;
        int activShift = j * activSize;

        float* srcPtr = layerRef[layerIndex - 1].outMat.data.mat + srcShift;
        float* dstPtr = layerRef[layerIndex].outMat.data.mat + dstShift;
        float* nbrPtr = layerRef[layerIndex].text.nbrUnroll.mat + nbrShift;
        float* ctrPtr = layerRef[layerIndex].text.ctrUnroll.mat + ctrShift;
        float* diffPtr = layerRef[layerIndex].text.diff.mat + diffShift;
        float* activPtr = layerRef[layerIndex].text.activ.mat + activShift;
        float* activBuf = layerRef[layerIndex].text.activBuf.mat + activShift;

        // Mapping
        for (int k = 0; k < nbrSize; k++)
        {
            nbrPtr[k] = srcPtr[nbrMap[k]];
        }

        for (int k = 0; k < ctrSize; k++)
        {
            ctrPtr[k] = srcPtr[ctrMap[k]];
        }

        // Find diff
        for (int row = 0; row < nbrRows; row++)
        {
            int nbrBase = row * nbrCols;
            for (int col = 0; col < nbrCols; col++)
            {
                int nbrShift = nbrBase + col;
                diffPtr[nbrShift] = nbrPtr[nbrShift] - ctrPtr[row];
            }
        }

        // Find activation
        cnn_activ_list[activId](activPtr, diffPtr, diffSize, activBuf);

        // Apply weight
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, dstImSize, chOut,
                    wCols, 1.0, activPtr, wCols, weight, wCols, 0.0, dstPtr,
                    dstImSize);

        // Add bias
        for (int ch = 0; ch < chOut; ch++)
        {
            cblas_saxpy(dstImSize, 1.0, &bias[ch], 0, dstPtr + ch * dstImSize,
                        1);
        }
    }
}

#endif
