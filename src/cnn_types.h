#ifndef __CNN_TYPES_H__
#define __CNN_TYPES_H__

#include "cnn.h"
#include "cnn_config.h"

#ifdef CNN_WITH_CUDA
#include <cudnn.h>
#endif

// CNN matrix type
struct CNN_MAT
{
    int rows;
    int cols;

    float* mat;
    float* grad;
};

// CNN shape type
struct CNN_SHAPE
{
    int width;
    int height;
    int channel;

    struct CNN_MAT data;
};

struct CNN_CONFIG_LAYER_INPUT
{
    // Layer type
    cnn_layer_t type;
};

struct CNN_CONFIG_LAYER_ACTIV
{
    // Layer type
    cnn_layer_t type;

    cnn_activ_t id;
};

struct CNN_CONFIG_LAYER_FC
{
    // Layer type
    cnn_layer_t type;

    int size;
};

struct CNN_CONFIG_LAYER_CONV
{
    // Layer type
    cnn_layer_t type;

    cnn_pad_t pad;
    cnn_dim_t dim;
    int size;
    int filter;
};

struct CNN_CONFIG_LAYER_POOL
{
    // Layer type
    cnn_layer_t type;

    cnn_pool_t poolType;
    cnn_dim_t dim;
    int size;
};

struct CNN_CONFIG_LAYER_DROP
{
    // Layer type
    cnn_layer_t type;

    float rate;
    float scale;
};

struct CNN_CONFIG_LAYER_BN
{
    // Layer type
    cnn_layer_t type;

    float rInit;
    float bInit;

    float expAvgFactor;
};

struct CNN_CONFIG_LAYER_TEXT
{
    // Layer type
    cnn_layer_t type;

    cnn_activ_t activId;
    int filter;
    float aInit;
};

union CNN_CONFIG_LAYER {
    // Layer type
    cnn_layer_t type;

    struct CNN_CONFIG_LAYER_INPUT input;
    struct CNN_CONFIG_LAYER_FC fc;
    struct CNN_CONFIG_LAYER_ACTIV activ;
    struct CNN_CONFIG_LAYER_CONV conv;
    struct CNN_CONFIG_LAYER_POOL pool;
    struct CNN_CONFIG_LAYER_DROP drop;
    struct CNN_CONFIG_LAYER_BN bn;
    struct CNN_CONFIG_LAYER_TEXT text;
};

struct CNN_CONFIG
{
    int width;
    int height;
    int channel;

    int batch;

    int layers;
    union CNN_CONFIG_LAYER* layerCfg;
};

struct CNN_LAYER_INPUT
{
    // Layer output matrix
    struct CNN_SHAPE outMat;
};

struct CNN_LAYER_ACTIV
{
    // Layer output matrix
    struct CNN_SHAPE outMat;

    // Gradient matrix
    struct CNN_MAT gradMat;

    // Calculate buffer
    struct CNN_MAT buf;
};

struct CNN_LAYER_FC
{
    // Layer output matrix
    struct CNN_SHAPE outMat;

    // Weight matrix
    struct CNN_MAT weight;

    // Bias vector
    struct CNN_MAT bias;

#ifdef CNN_WITH_CUDA
    cudnnReduceTensorDescriptor_t reduDesc;

    cudnnTensorDescriptor_t biasTen;
    cudnnTensorDescriptor_t dstTen;

    size_t indSize;
    uint32_t* indData;
#endif
};

struct CNN_LAYER_CONV
{
    // Layer output matrix
    struct CNN_SHAPE outMat;

    // Kernel matrix
    struct CNN_MAT kernel;

    // Bias vector
#if defined(CNN_CONV_BIAS_FILTER) || defined(CNN_CONV_BIAS_LAYER)
    struct CNN_MAT bias;
#endif

    // Channel
    int inChannel;

#ifdef CNN_WITH_CUDA
    cudnnConvolutionDescriptor_t convDesc;

    cudnnTensorDescriptor_t srcTen;
    cudnnTensorDescriptor_t dstTen;
    cudnnFilterDescriptor_t kernelTen;

#if defined(CNN_CONV_BIAS_FILTER)
    cudnnTensorDescriptor_t biasTen;
#elif defined(CNN_CONV_BIAS_LAYER)
#error Unsupported convolution bias type
#endif

    cudnnConvolutionFwdAlgo_t convAlgoFW;
    cudnnConvolutionBwdFilterAlgo_t convAlgoBWFilter;
    cudnnConvolutionBwdDataAlgo_t convAlgoBWGrad;

#else
    // Convolution to gemm
    int* indexMap;
    struct CNN_MAT unroll;
#endif
};

struct CNN_LAYER_POOL
{
    // Layer output matrix
    struct CNN_SHAPE outMat;

#ifdef CNN_WITH_CUDA
    cudnnPoolingDescriptor_t poolDesc;

    cudnnTensorDescriptor_t srcTen;
    cudnnTensorDescriptor_t dstTen;
#else
    // Pooling index
    int* indexMat;
#endif
};

struct CNN_LAYER_DROP
{
    // Layer output matrix
    struct CNN_SHAPE outMat;

    // Dropout mask
    int* mask;
    int* maskGpu;
};

struct CNN_LAYER_BN
{
    // Layer output matrix
    struct CNN_SHAPE outMat;

    // BatchNorm variables
    struct CNN_MAT bnScale;
    struct CNN_MAT bnBias;

    struct CNN_MAT saveMean;
    struct CNN_MAT saveVar;

    struct CNN_MAT runMean;
    struct CNN_MAT runVar;

#ifdef CNN_WITH_CUDA
    cudnnTensorDescriptor_t srcTen;
    cudnnTensorDescriptor_t bnTen;
#endif
};

struct CNN_LAYER_TEXT
{
    // Layer output matrix
    struct CNN_SHAPE outMat;

    // Texture weights
    struct CNN_MAT weight;
    struct CNN_MAT bias;
    struct CNN_MAT alpha;

    // Channel
    int inChannel;

    // Texture calculation buffer
    struct CNN_MAT nbrUnroll;
    struct CNN_MAT ctrUnroll;
    struct CNN_MAT diff;
    struct CNN_MAT scale;
    struct CNN_MAT activ;

    // Index mapping
    int* nbrMap;  // Neighbor mapping
    int* ctrMap;  // Center mapping
};

union CNN_LAYER {
    // Layer output matrix
    struct CNN_SHAPE outMat;

    struct CNN_LAYER_INPUT input;
    struct CNN_LAYER_FC fc;
    struct CNN_LAYER_ACTIV activ;
    struct CNN_LAYER_CONV conv;
    struct CNN_LAYER_POOL pool;
    struct CNN_LAYER_DROP drop;
    struct CNN_LAYER_BN bn;
    struct CNN_LAYER_TEXT text;
};

struct CNN
{
    struct CNN_CONFIG cfg;
    union CNN_LAYER* layerList;

    cnn_opmode_t opMode;
};

#endif
