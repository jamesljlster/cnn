#ifndef __CNN_XML_H__
#define __CNN_XML_H__

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
		goto errLabel; \
	}
#else
#define cnn_xml_run(func, retVal, errLabel) \
	retVal = func; \
	if(retVal < 0) \
	{ \
		goto errLabel; \
	}
#endif

#endif
