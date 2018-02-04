#ifndef __CNN_PRIVATE_H__
#define __CNN_PRIVATE_H__

#include "cnn.h"
#include "cnn_types.h"

// Macros
#ifdef DEBUG
#include <stdio.h>
#define cnn_free(ptr)	fprintf(stderr, "%s(): free(%s), %p\n", __FUNCTION__, #ptr, ptr); free(ptr)
#else
#define cnn_free(ptr)	free(ptr)
#endif

#define cnn_alloc(ptr, len, type, retVar, errLabel) \
	ptr = calloc(len, sizeof(type)); \
	if(ptr == NULL) \
	{ \
		ret = CNN_MEM_FAILED; \
		goto errLabel; \
	}

#ifdef DEBUG
#define cnn_run(func, retVal, errLabel) \
	retVal = func; \
	if(retVal != CNN_NO_ERROR) \
	{ \
		fprintf(stderr, "%s(): %s failed with error: %d\n", __FUNCTION__, #func, retVal); \
		goto errLabel; \
	}
#else
#define cnn_run(func, retVal, errLabel) \
	retVal = func; \
	if(retVal != CNN_NO_ERROR) \
	{ \
		goto errLabel; \
	}
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Private config functions
int cnn_config_init(cnn_config_t cfg);
void cnn_config_struct_delete(struct CNN_CONFIG* cfg);
int cnn_config_struct_clone(struct CNN_CONFIG* dstPtr, const struct CNN_CONFIG* src);

// Private allocate functions
int cnn_mat_alloc(struct CNN_MAT* matPtr, int rows, int cols, int needGrad);

int cnn_layer_input_alloc(union CNN_LAYER* layerPtr,
		int inWidth, int inHeight, int batch);
int cnn_layer_afunc_alloc(struct CNN_LAYER_AFUNC* layerPtr,
		int inWidth, int inHeight, int batch, int aFuncID);
int cnn_layer_fc_alloc(struct CNN_LAYER_FC* layerPtr,
		int inWidth, int inHeight, int outSize, int batch);
int cnn_layer_conv_alloc(struct CNN_LAYER_CONV* layerPtr,
		int inWidth, int inHeight, int size, int batch);
int cnn_layer_pool_alloc(struct CNN_LAYER_POOL* layerPtr,
		int inWidth, int inHeight, int size, int batch);

int cnn_network_alloc(struct CNN* cnn, const struct CNN_CONFIG* cfg);

// Private delete functions
void cnn_mat_delete(struct CNN_MAT* matPtr);

void cnn_layer_input_delete(union CNN_LAYER* layerPtr);
void cnn_layer_afunc_delete(struct CNN_LAYER_AFUNC* layerPtr);
void cnn_layer_fc_delete(struct CNN_LAYER_FC* layerPtr);
void cnn_layer_conv_delete(struct CNN_LAYER_CONV* layerPtr);
void cnn_layer_pool_delete(struct CNN_LAYER_POOL* layerPtr);

void cnn_network_delete(struct CNN* cnn);
void cnn_struct_delete(struct CNN* cnn);

#ifdef __cplusplus
}
#endif

#endif
