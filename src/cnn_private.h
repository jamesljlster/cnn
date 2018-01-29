#ifndef __CNN_PRIVATE_H__
#define __CNN_PRIVATE_H__

#include "cnn.h"
#include "cnn_types.h"

// Macros
#define cnn_alloc(ptr, len, type, retVar, errLabel) \
	ptr = calloc(len, sizeof(type)); \
	if(ptr == NULL) \
	{ \
		ret = LSTM_MEM_FAILED; \
		goto errLabel; \
	}

#ifdef DEBUG
#define cnn_run(func, retVal, errLabel) \
	retVal = func; \
	if(retVal != LSTM_NO_ERROR) \
	{ \
		fprintf(stderr, "%s(): %s failed with error: %d\n", __FUNCTION__, #func, retVal); \
		goto errLabel; \
	}
#else
#define cnn_run(func, retVal, errLabel) \
	retVal = func; \
	if(retVal != LSTM_NO_ERROR) \
	{ \
		goto errLabel; \
	}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif
