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

int cnn_write_config_xml(struct CNN_CONFIG* cfgRef, xmlTextWriterPtr writer);
int cnn_write_network_xml(struct CNN_CONFIG* cfgRef, union CNN_LAYER* layerRef,
		xmlTextWriterPtr writer);

int cnn_write_layer_input_xml(xmlTextWriterPtr writer);
int cnn_write_layer_fc_xml(struct CNN_CONFIG* cfgRef, union CNN_LAYER* layerRef,
		int layerIndex, xmlTextWriterPtr writer);
int cnn_write_layer_afunc_xml(struct CNN_CONFIG* cfgRef, int layerIndex,
		xmlTextWriterPtr writer);
int cnn_write_layer_conv_xml(struct CNN_CONFIG* cfgRef, union CNN_LAYER* layerRef,
		int layerIndex, xmlTextWriterPtr writer);
int cnn_write_layer_pool_xml(struct CNN_CONFIG* cfgRef, int layerIndex,
		xmlTextWriterPtr writer);

int cnn_write_dim_attr_xml(int dim, xmlTextWriterPtr writer);

int cnn_write_mat_xml(struct CNN_MAT* matPtr, const char* nodeName, xmlTextWriterPtr writer);

#ifdef __cplusplus
}
#endif

#endif
