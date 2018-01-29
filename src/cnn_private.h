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

#ifdef __cplusplus
}
#endif

#endif
