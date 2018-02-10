#ifndef __CNN_WRITE_H__
#define __CNN_WRITE_H__

#include "cnn.h"
#include "cnn_private.h"
#include "cnn_xml.h"

inline void cnn_ftostr(char* buf, int bufSize, float val)
{
	int __ret = snprintf(buf, bufSize, "%.32g", val);
	assert(__ret > 0 && __ret <= bufSize && "Insufficient buffer size");
}

inline void cnn_itostr(char* buf, int bufSize, int val)
{
	int __ret = snprintf(buf, bufSize, "%d", val);
	assert(__ret > 0 && __ret <= bufSize && "Insufficient buffer size");
}

#ifdef __cplusplus
extern "C" {
#endif

int cnn_export_root(struct CNN_CONFIG* cfgRef, union CNN_LAYER* layerRef, const char* fPath);

#ifdef __cplusplus
}
#endif

#endif
