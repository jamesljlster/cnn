#ifndef __CNN_TEXT_H__
#define __CNN_TEXT_H__

#include <string.h>

#include <cblas.h>

#include "cnn_builtin_math.h"
#include "cnn_types.h"

#ifdef CNN_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cnn_init.h"

void cnn_text_map_gpu(float* dst, float* src, int* map, int len);
void cnn_text_map_inv_gpu(float* dst, float* src, int* map, int len);
void cnn_text_find_diff_gpu(float* diffPtr, float* nbrPtr, float* ctrPtr,
                            int nbrRows, int nbrCols);
void cnn_text_find_diff_grad_gpu(float* gradIn, float* diffGrad, float* nbrGrad,
                                 float* ctrGrad, int nbrRows, int nbrCols);
#endif

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

    int nbrRows = dstImSize * chIn;
    int nbrCols = wSize;
    int nbrSize = nbrRows * nbrCols;

    int ctrRows = nbrRows;
    int ctrCols = 1;
    int ctrSize = ctrRows * ctrCols;

    int diffRows = dstImSize;
    int diffCols = wCols;
    int diffSize = diffRows * diffCols;

    int scaleRows = dstImSize;
    int scaleCols = wCols;
    int scaleSize = scaleRows * scaleCols;

    int activRows = dstImSize;
    int activCols = wCols;
    int activSize = activRows * activCols;

    int* nbrMap = layerRef[layerIndex].text.nbrMap;
    int* ctrMap = layerRef[layerIndex].text.ctrMap;
    float* alpha = layerRef[layerIndex].text.alpha.mat;
    float* weight = layerRef[layerIndex].text.weight.mat;
    float* bias = layerRef[layerIndex].text.bias.mat;

    for (int j = 0; j < cfgRef->batch; j++)
    {
        int srcShift = j * srcSize;
        int dstShift = j * dstSize;
        int nbrShift = j * nbrSize;
        int ctrShift = j * ctrSize;
        int diffShift = j * diffSize;
        int scaleShift = j * scaleSize;
        int activShift = j * activSize;

        float* srcPtr = layerRef[layerIndex - 1].outMat.data.mat + srcShift;
        float* dstPtr = layerRef[layerIndex].outMat.data.mat + dstShift;
        float* nbrPtr = layerRef[layerIndex].text.nbrUnroll.mat + nbrShift;
        float* ctrPtr = layerRef[layerIndex].text.ctrUnroll.mat + ctrShift;
        float* diffPtr = layerRef[layerIndex].text.diff.mat + diffShift;
        float* scalePtr = layerRef[layerIndex].text.scale.mat + scaleShift;
        float* activPtr = layerRef[layerIndex].text.activ.mat + activShift;

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

        // Find scaled diff
        for (int row = 0; row < diffRows; row++)
        {
            int diffBase = row * diffCols;
            for (int c = 0; c < chIn; c++)
            {
                int diffChBase = diffBase + c * wSize;
                for (int w = 0; w < wSize; w++)
                {
                    int diffShift = diffChBase + w;
                    scalePtr[diffShift] = diffPtr[diffShift] * alpha[c];
                }
            }
        }

        // Find activation
        cnn_activ_list[activId](activPtr, scalePtr, scaleSize, NULL);

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

static inline void cnn_backward_text(union CNN_LAYER* layerRef,
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

    int nbrRows = dstImSize * chIn;
    int nbrCols = wSize;
    int nbrSize = nbrRows * nbrCols;

    int ctrRows = nbrRows;
    int ctrCols = 1;
    int ctrSize = ctrRows * ctrCols;

    int diffRows = dstImSize;
    int diffCols = wCols;
    int diffSize = diffRows * diffCols;

    int scaleRows = dstImSize;
    int scaleCols = wCols;
    int scaleSize = scaleRows * scaleCols;

    int activRows = dstImSize;
    int activCols = wCols;
    int activSize = activRows * activCols;

    int* nbrMap = layerRef[layerIndex].text.nbrMap;
    int* ctrMap = layerRef[layerIndex].text.ctrMap;
    float* alpha = layerRef[layerIndex].text.alpha.mat;
    float* aGrad = layerRef[layerIndex].text.alpha.grad;
    float* weight = layerRef[layerIndex].text.weight.mat;
    float* wGrad = layerRef[layerIndex].text.weight.grad;
    float* bGrad = layerRef[layerIndex].text.bias.grad;

    for (int j = 0; j < cfgRef->batch; j++)
    {
        int dstShift = j * dstSize;
        int activShift = j * activSize;
        int scaleShift = j * scaleSize;
        int diffShift = j * diffSize;

        float* gradPtr = layerRef[layerIndex].outMat.data.grad + dstShift;
        float* activPtr = layerRef[layerIndex].text.activ.mat + activShift;
        float* activGrad = layerRef[layerIndex].text.activ.grad + activShift;
        float* scalePtr = layerRef[layerIndex].text.scale.mat + scaleShift;
        float* scaleGrad = layerRef[layerIndex].text.scale.grad + scaleShift;
        float* diffPtr = layerRef[layerIndex].text.diff.mat + diffShift;

        // Sum weight gradient matrix
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, activCols, chOut,
                    activRows, 1.0, activPtr, activCols, gradPtr, activRows,
                    1.0, wGrad, activCols);

        // Sum bias gradient matrix
        for (int ch = 0; ch < chOut; ch++)
        {
            cblas_saxpy(activRows, 1.0, gradPtr + ch * activRows, 1, bGrad + ch,
                        0);
        }

        // Sum activation gradient matrix
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, activCols,
                    activRows, chOut, 1.0, weight, activCols, gradPtr,
                    activRows, 0.0, activGrad, activCols);

        // Find scale gradient matrix
        cnn_activ_grad_list[activId](scaleGrad, scalePtr, NULL, scaleSize,
                                     scaleGrad, NULL);
        for (int i = 0; i < scaleSize; i++)
        {
            scaleGrad[i] *= activGrad[i];
        }

        // Sum alpha gradient
        for (int row = 0; row < diffRows; row++)
        {
            int diffBase = row * diffCols;
            for (int c = 0; c < chIn; c++)
            {
                int diffChBase = diffBase + c * wSize;
                for (int w = 0; w < wSize; w++)
                {
                    int diffShift = diffChBase + w;
                    aGrad[c] += scaleGrad[diffShift] * diffPtr[diffShift];
                }
            }
        }
    }

    // Find layer gradient
    if (layerIndex > 1)
    {
        // Clear gradient
        memset(layerRef[layerIndex].text.ctrUnroll.grad, 0,
               sizeof(float) * layerRef[layerIndex].text.ctrUnroll.rows *
                   layerRef[layerIndex].text.ctrUnroll.cols);
        memset(layerRef[layerIndex - 1].outMat.data.grad, 0,
               sizeof(float) * layerRef[layerIndex - 1].outMat.data.rows *
                   layerRef[layerIndex - 1].outMat.data.cols);

        for (int j = 0; j < cfgRef->batch; j++)
        {
            int srcShift = j * srcSize;
            int nbrShift = j * nbrSize;
            int ctrShift = j * ctrSize;
            int diffShift = j * diffSize;
            int scaleShift = j * scaleSize;

            float* preGradPtr =
                layerRef[layerIndex - 1].outMat.data.grad + srcShift;
            float* nbrGrad =
                layerRef[layerIndex].text.nbrUnroll.grad + nbrShift;
            float* ctrGrad =
                layerRef[layerIndex].text.ctrUnroll.grad + ctrShift;
            float* diffGrad = layerRef[layerIndex].text.diff.grad + diffShift;
            float* scaleGrad =
                layerRef[layerIndex].text.scale.grad + scaleShift;

            // Sum diff gradient matrix
            for (int row = 0; row < diffRows; row++)
            {
                int diffBase = row * diffCols;
                for (int c = 0; c < chIn; c++)
                {
                    int diffChBase = diffBase + c * wSize;
                    for (int w = 0; w < wSize; w++)
                    {
                        int diffShift = diffChBase + w;
                        diffGrad[diffShift] = scaleGrad[diffShift] * alpha[c];
                    }
                }
            }

            // Sum neighbor, center gradient matrix
            for (int row = 0; row < nbrRows; row++)
            {
                int nbrBase = row * nbrCols;
                for (int col = 0; col < nbrCols; col++)
                {
                    int nbrShift = nbrBase + col;
                    float gradTmp = diffGrad[nbrShift];
                    nbrGrad[nbrShift] = gradTmp;
                    ctrGrad[row] -= gradTmp;
                }
            }

            // Sum layer gradient matrix
            for (int k = 0; k < nbrSize; k++)
            {
                preGradPtr[nbrMap[k]] += nbrGrad[k];
            }

            for (int k = 0; k < ctrSize; k++)
            {
                preGradPtr[ctrMap[k]] += ctrGrad[k];
            }
        }
    }
}

#endif
