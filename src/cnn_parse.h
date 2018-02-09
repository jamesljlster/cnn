#ifndef __CNN_PARSE_H__
#define __CNN_PARSE_H__

#include <libxml/parser.h>
#include <libxml/xpath.h>

#include "cnn.h"
#include "cnn_private.h"

#ifdef __cplusplus
extern "C" {
#endif

int cnn_parse_config_xml(struct CNN_CONFIG* cfgRef, xmlNodePtr node);
int cnn_parse_config_afunc_xml(struct CNN_CONFIG* cfgRef, xmlNodePtr node);
int cnn_parse_config_layer_xml(struct CNN_CONFIG* cfgRef, xmlNodePtr node);

#ifdef __cplusplus
}
#endif

#endif
