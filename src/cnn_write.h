#ifndef __CNN_WRITE_H__
#define __CNN_WRITE_H__

#include "cnn.h"
#include "cnn_private.h"
#include "cnn_xml.h"

#ifdef __cplusplus
extern "C" {
#endif

int cnn_write_config_xml(struct CNN_CONFIG* cfgRef, xmlTextWriterPtr writer);

#ifdef __cplusplus
}
#endif

#endif
