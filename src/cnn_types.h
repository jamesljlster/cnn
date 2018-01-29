#ifndef __CNN_TYPES_H__
#define __CNN_TYPES_H__

// CNN matrix type
struct CNN_MAT
{
	int width;
	int height;

	int rows;
	int cols;

	float* mat;
	float* grad;
};

struct CNN_CONFIG_LAYER_AFUNC
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

	int dim;
	int size;
};

union CNN_CONFIG_LAYER
{
	// Layer type
	int type;

	struct CNN_CONFIG_LAYER_AFUNC aFunc;
	struct CNN_CONFIG_LAYER_FC fc;
	struct CNN_CONFIG_LAYER_CONV conv;
};

struct CNN_CONFIG
{
	int width;
	int height;
	int outputs;

	int batch;

	float lRate;

	int layers;
	union CNN_CONFIG_LAYER* layerCfg;
};

struct CNN_LAYER_AFUNC
{
	// Layer output matrix
	struct CNN_MAT outMat;

	// Gradient matrix
	struct CNN_MAT gradMat;
};

struct CNN_LAYER_FC
{
	// Layer output matrix
	struct CNN_MAT outMat;

	// Weight matrix
	struct CNN_MAT weight;

	// Bias vector
	struct CNN_MAT bias;
};

struct CNN_LAYER_CONV
{
	// Layer output matrix
	struct CNN_MAT outMat;

	// Kernel matrix
	struct CNN_MAT kernel;
};

union CNN_LAYER
{
	// Layer output matrix
	struct CNN_MAT outMat;

	struct CNN_LAYER_AFUNC aFunc;
	struct CNN_LAYER_FC fc;
	struct CNN_LAYER_CONV conv;
};

struct CNN
{
	struct CNN_CONFIG cfg;
	union CNN_LAYER* layerList;
};

#endif
