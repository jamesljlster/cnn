#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "cnn_parse.h"
#include "cnn_strdef.h"

int cnn_parse_network_xml(struct CNN_CONFIG* cfgPtr, union CNN_LAYER** layerPtr,
		xmlNodePtr node)
{
	int ret = CNN_NO_ERROR;

RET:
	return ret;
}

int cnn_parse_config_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node)
{
	int ret = CNN_NO_ERROR;
	int strId;

	xmlNodePtr cur;
	xmlNodePtr input; // Must exist
	xmlNodePtr lRate, batch;

	// Find node
	input = NULL;
	lRate = NULL;
	batch = NULL;
	cur = node->xmlChildrenNode;
	while(cur != NULL)
	{
		// Find node id
		strId = cnn_strdef_get_id((const char*)cur->name);
		switch(strId)
		{
			case CNN_STR_INPUT:
				input = cur;
				break;

			case CNN_STR_BATCH:
				batch = cur;
				break;

			case CNN_STR_LRATE:
				lRate = cur;
				break;
		}

		cur = cur->next;
	}

	if(input == NULL)
	{
		ret = CNN_INFO_NOT_FOUND;
		goto RET;
	}

	// Parse input
	cnn_run(cnn_parse_config_input_xml(cfgPtr, input), ret, RET);

	// Parse batch
	if(batch != NULL)
	{
		cnn_run(cnn_strtoi(&cfgPtr->batch, (const char*)xmlNodeGetContent(batch)), ret, RET);
	}

	// Parse learning rate
	if(lRate != NULL)
	{
		cnn_run(cnn_strtof(&cfgPtr->lRate, (const char*)xmlNodeGetContent(lRate)), ret, RET);
	}

RET:
	return ret;
}

int cnn_parse_config_input_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node)
{
	int ret = CNN_NO_ERROR;
	int strId;

	xmlNodePtr cur;
	xmlNodePtr width, height, channel;

	// Find node
	width = NULL;
	height = NULL;
	channel = NULL;
	cur = node->xmlChildrenNode;
	while(cur != NULL)
	{
		// Find node id
		strId = cnn_strdef_get_id((const char*)cur->name);
		switch(strId)
		{
			case CNN_STR_WIDTH:
				width = cur;
				break;

			case CNN_STR_HEIGHT:
				height = cur;
				break;

			case CNN_STR_CHANNEL:
				channel = cur;
				break;
		}

		cur = cur->next;
	}

	if(width == NULL || height == NULL || channel == NULL)
	{
		ret = CNN_INFO_NOT_FOUND;
		goto RET;
	}

	// Parse information
	cnn_run(cnn_strtoi(&cfgPtr->width, (const char*)xmlNodeGetContent(width)), ret, RET);
	cnn_run(cnn_strtoi(&cfgPtr->height, (const char*)xmlNodeGetContent(height)), ret, RET);
	cnn_run(cnn_strtoi(&cfgPtr->channel, (const char*)xmlNodeGetContent(channel)), ret, RET);

RET:
	return ret;
}

int cnn_import_root(struct CNN_CONFIG* cfgPtr, union CNN_LAYER** layerPtr, const char* fPath)
{
	int ret = CNN_NO_ERROR;
	int strId;

	xmlDocPtr doc = NULL;
	xmlNodePtr cur, cfg, net;

	// Parse xml
	doc = xmlParseFile(fPath);
	if(doc == NULL)
	{
		ret = CNN_FILE_OP_FAILED;
		goto RET;
	}

	// Check xml file
	cur = xmlDocGetRootElement(doc);
	if(cur == NULL)
	{
		ret = CNN_INVALID_FILE;
		goto RET;
	}

	if(xmlStrcmp(cur->name, (xmlChar*)cnn_str_list[CNN_STR_MODEL]))
	{
		ret = CNN_INVALID_FILE;
		goto RET;
	}

	// Find config and network node
	cfg = NULL;
	net = NULL;
	cur = cur->xmlChildrenNode;
	while(cur != NULL)
	{
		// Find node id
		strId = cnn_strdef_get_id((const char*)cur->name);
		switch(strId)
		{
			case CNN_STR_CONFIG:
				cfg = cur;
				break;

			case CNN_STR_NETWORK:
				net = cur;
				break;
		}

		cur = cur->next;
	}

	if(cfg == NULL || net == NULL)
	{
		ret = CNN_INFO_NOT_FOUND;
		goto RET;
	}

	// Parse config
	cnn_run(cnn_parse_config_xml(cfgPtr, cfg), ret, RET);

	// Parse network
	cnn_run(cnn_parse_network_xml(cfgPtr, layerPtr, cfg), ret, RET);

RET:
	if(doc != NULL)
	{
		xmlFreeDoc(doc);
	}

	return ret;
}
