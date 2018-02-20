#ifndef __CNN_TYPES_H__
#define __CNN_TYPES_H__

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
};

union CNN_CONFIG_LAYER
{
	// Layer type
	int type;

	struct CNN_CONFIG_LAYER_AFUNC aFunc;
	struct CNN_CONFIG_LAYER_FC fc;
	struct CNN_CONFIG_LAYER_CONV conv;
	struct CNN_CONFIG_LAYER_POOL pool;
	struct CNN_CONFIG_LAYER_DROP drop;
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

struct CNN_LAYER_AFUNC
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
	struct CNN_MAT bias;

	// Channel
	int inChannel;
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
	struct CNN_MAT mask;
};

union CNN_LAYER
{
	// Layer output matrix
	struct CNN_SHAPE outMat;

	struct CNN_LAYER_AFUNC aFunc;
	struct CNN_LAYER_FC fc;
	struct CNN_LAYER_CONV conv;
	struct CNN_LAYER_POOL pool;
	struct CNN_LAYER_DROP drop;
};

struct CNN
{
	struct CNN_CONFIG cfg;
	union CNN_LAYER* layerList;
};

#endif
