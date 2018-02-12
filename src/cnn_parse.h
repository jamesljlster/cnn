#ifndef __CNN_PARSE_H__
#define __CNN_PARSE_H__

#include "cnn.h"
#include "cnn_private.h"
#include "cnn_xml.h"

inline int cnn_strtoi(int* valPtr, const char* str)
{
	char* __ptr;
	int __tmp;
	if(str == NULL)
	{
		return CNN_INFO_NOT_FOUND;
	}
	__tmp = strtol(str, &__ptr, 10);
	if(__ptr == str)
	{
		return CNN_PARSE_FAILED;
	}
	else
	{
		*valPtr = __tmp;
		return CNN_NO_ERROR;
	}
}

inline int cnn_strtof(float* valPtr, const char* str)
{
	char* __ptr;
	float __tmp;
	if(str == NULL)
	{
		return CNN_INFO_NOT_FOUND;
	}
	__tmp = strtod(str, &__ptr);
	if(__ptr == str)
	{
		return CNN_PARSE_FAILED;
	}
	else
	{
		*valPtr = __tmp;
		return CNN_NO_ERROR;
	}
}

#ifdef __cplusplus
extern "C" {
#endif

int cnn_import_root(struct CNN_CONFIG* cfgPtr, union CNN_LAYER** layerPtr, const char* fPath);

int cnn_parse_config_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node);
int cnn_parse_config_input_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node);

int cnn_parse_network_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node);
int cnn_parse_network_layer_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node);

#ifdef __cplusplus
}
#endif

#endif
