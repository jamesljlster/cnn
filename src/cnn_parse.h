#ifndef __CNN_PARSE_H__
#define __CNN_PARSE_H__

#include "cnn.h"
#include "cnn_private.h"
#include "cnn_xml.h"

static inline int cnn_strtoi(int* valPtr, const char* str)
{
    char* __ptr;
    int __tmp;
    if (str == NULL)
    {
        return CNN_INFO_NOT_FOUND;
    }
    __tmp = strtol(str, &__ptr, 10);
    if (__ptr == str)
    {
        return CNN_PARSE_FAILED;
    }
    else
    {
        *valPtr = __tmp;
        return CNN_NO_ERROR;
    }
}

static inline int cnn_strtof(float* valPtr, const char* str)
{
    char* __ptr;
    float __tmp;
    if (str == NULL)
    {
        return CNN_INFO_NOT_FOUND;
    }
    __tmp = strtod(str, &__ptr);
    if (__ptr == str)
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
extern "C"
{
#endif

    int cnn_import_root(struct CNN_CONFIG* cfgPtr, union CNN_LAYER** layerPtr,
                        const char* fPath);

    int cnn_parse_config_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node);
    int cnn_parse_config_input_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node);

    int cnn_parse_network_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node);
    int cnn_parse_network_layer_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node);

    int cnn_parse_network_detail_xml(struct CNN* cnn, xmlDocPtr doc);
    int cnn_parse_network_detail_fc_xml(struct CNN* cnn, int layerIndex,
                                        xmlNodePtr node);
    int cnn_parse_network_detail_conv_xml(struct CNN* cnn, int layerIndex,
                                          xmlNodePtr node);
    int cnn_parse_network_detail_bn_xml(struct CNN* cnn, int layerIndex,
                                        xmlNodePtr node);
    int cnn_parse_network_detail_text_xml(struct CNN* cnn, int layerIndex,
                                          xmlNodePtr node);
    int cnn_parse_mat(struct CNN_MAT* mat, xmlNodePtr node);

#ifdef __cplusplus
}
#endif

#endif
