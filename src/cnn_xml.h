#ifndef __CNN_XML_H__
#define __CNN_XML_H__

#include <assert.h>

#include <libxml/xmlwriter.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>

#ifdef DEBUG
#define cnn_xml_run(func, retVal, errLabel) \
	retVal = func; \
	if(retVal < 0) \
	{ \
		fprintf(stderr, "%s(), %d: %s failed with error: %d\n", __FUNCTION__, __LINE__, \
			#func, retVal); \
		ret = CNN_FILE_OP_FAILED; \
		goto errLabel; \
	} \
	else \
	{ \
		retVal = CNN_NO_ERROR; \
	}
#else
#define cnn_xml_run(func, retVal, errLabel) \
	retVal = func; \
	if(retVal < 0) \
	{ \
		ret = CNN_FILE_OP_FAILED; \
		goto errLabel; \
	} \
	else \
	{ \
		retVal = CNN_NO_ERROR; \
	}
#endif

#define CNN_XML_BUFLEN 64

#define CNN_XML_VER_STR "1.0"
#define CNN_XML_ENC_STR "utf-8"

#endif
