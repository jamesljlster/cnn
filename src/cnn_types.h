#ifndef __CNN_TYPES_H__
#define __CNN_TYPES_H__

#include "cnn_config.h"

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
    int type;
};

struct CNN_CONFIG_LAYER_ACTIV
{
    // Layer type
    int type;

    int id;
};

struct CNN_CONFIG_LAYER_FC
{
    // Layer type
    int type;

    int size;
};

struct CNN_CONFIG_LAYER_CONV
{
    // Layer type
    int type;

    int pad;
    int dim;
    int size;
    int filter;
};

struct CNN_CONFIG_LAYER_POOL
{
    // Layer type
    int type;

    int poolType;
    int dim;
    int size;
};

struct CNN_CONFIG_LAYER_DROP
{
    // Layer type
    int type;

    float rate;
    float scale;
};

struct CNN_CONFIG_LAYER_BN
{
    // Layer type
    int type;

    float rInit;
    float bInit;
};

struct CNN_CONFIG_LAYER_TEXT
{
    // Layer type
    int type;

    int activId;
    int filter;
};

union CNN_CONFIG_LAYER {
    // Layer type
    int type;

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

    float lRate;

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

    // Convolution to gemm
    int* indexMap;
    struct CNN_MAT unroll;
};

struct CNN_LAYER_POOL
{
    // Layer output matrix
    struct CNN_SHAPE outMat;

    // Pooling index
    int* indexMat;
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
    struct CNN_MAT bnVar;

    // Cache
    float* stddev;
    struct CNN_MAT srcShift;
    struct CNN_MAT srcNorm;

#ifdef CNN_WITH_CUDA
    // Buffer
    float* buf;
#endif
};

struct CNN_LAYER_TEXT
{
    // Layer output matrix
    struct CNN_SHAPE outMat;

    // Texture weights
    struct CNN_MAT weight;
    struct CNN_MAT bias;

    // Channel
    int inChannel;

    // Texture calculation buffer
    struct CNN_MAT nbrUnroll;
    struct CNN_MAT ctrUnroll;
    struct CNN_MAT diff;
    struct CNN_MAT activ;
    struct CNN_MAT activBuf;

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

    int dropEnable;
};

#endif
