#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn_parse.h"
#include "cnn_strdef.h"
#include "cnn_builtin_math.h"

int cnn_parse_network_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node)
{
	int ret = CNN_NO_ERROR;
	int strId;
	int tmp;

	xmlNodePtr cur;
	xmlAttrPtr attrCur, layers;

	// Find layers
	layers = NULL;
	attrCur = node->properties;
	while(attrCur != NULL)
	{
		// Find attr id
		strId = cnn_strdef_get_id((const char*)attrCur->name);
		switch(strId)
		{
			case CNN_STR_SIZE:
				layers = attrCur;
				break;
		}

		attrCur = attrCur->next;
	}

	if(layers == NULL)
	{
		ret = CNN_INFO_NOT_FOUND;
		goto RET;
	}

	// Parse and set layers
	cnn_run(cnn_strtoi(&tmp, (const char*)xmlNodeGetContent(layers->children)), ret, RET);
	cnn_run(cnn_config_set_layers(cfgPtr, tmp), ret, RET);

	cur = node->xmlChildrenNode;
	while(cur != NULL)
	{
		// Find node id
		strId = cnn_strdef_get_id((const char*)cur->name);
		switch(strId)
		{
			case CNN_STR_LAYER:
				cnn_run(cnn_parse_network_layer_xml(cfgPtr, cur), ret, RET);
				break;
		}

		cur = cur->next;
	}

RET:
	return ret;
}

int cnn_parse_network_layer_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node)
{
	int ret = CNN_NO_ERROR;
	int tmpType, tmpIndex;
	int strId;

	xmlAttrPtr attrCur;
	xmlAttrPtr index = NULL, type = NULL;
	xmlAttrPtr dim = NULL, size = NULL, poolType = NULL, id = NULL;

	// Parse attribute
	attrCur = node->properties;
	while(attrCur != NULL)
	{
		// Find node id
		strId = cnn_strdef_get_id((const char*)attrCur->name);
		switch(strId)
		{
			case CNN_STR_INDEX:
				index = attrCur;
				break;

			case CNN_STR_TYPE:
				type = attrCur;
				break;

			case CNN_STR_DIM:
				dim = attrCur;
				break;

			case CNN_STR_POOL_TYPE:
				poolType = attrCur;
				break;

			case CNN_STR_ID:
				id = attrCur;
				break;

			case CNN_STR_SIZE:
				size = attrCur;
				break;
		}

		attrCur = attrCur->next;
	}

	if(index == NULL || type == NULL)
	{
		ret = CNN_INFO_NOT_FOUND;
		goto RET;
	}

	// Parse type and index
	cnn_run(cnn_strtoi(&tmpIndex, (const char*)xmlNodeGetContent(index->children)), ret, RET);
	strId = cnn_strdef_get_id((const char*)xmlNodeGetContent(type->children));
	switch(strId)
	{
		case CNN_STR_INPUT:
			tmpType = CNN_LAYER_INPUT;
			break;

		case CNN_STR_FC:
			tmpType = CNN_LAYER_FC;
			break;

		case CNN_STR_AFUNC:
			tmpType = CNN_LAYER_AFUNC;
			break;

		case CNN_STR_CONV:
			tmpType = CNN_LAYER_CONV;
			break;

		case CNN_STR_POOL:
			tmpType = CNN_LAYER_POOL;
			break;

		default:
			assert(!"Invalid layer type");
	}

	// Parse layer
	cfgPtr->layerCfg[tmpIndex].type = tmpType;
	switch(tmpType)
	{
		case CNN_LAYER_INPUT:
			break;

		case CNN_LAYER_FC:
			if(size == NULL)
			{
				ret = CNN_INFO_NOT_FOUND;
				goto RET;
			}

			// Parse size
			cnn_run(cnn_strtoi(&cfgPtr->layerCfg[tmpIndex].fc.size,
						(const char*)xmlNodeGetContent(size->children)),
					ret, RET);
			break;

		case CNN_LAYER_AFUNC:
			if(id == NULL)
			{
				ret = CNN_INFO_NOT_FOUND;
				goto RET;
			}

			// Parse id
			strId = cnn_get_afunc_id((const char*)xmlNodeGetContent(id->children));
			assert(strId >= 0 && "Invalid activation function ID");
			cfgPtr->layerCfg[tmpIndex].aFunc.id = strId;

			break;

		case CNN_LAYER_CONV:
			if(dim == NULL || size == NULL)
			{
				ret = CNN_INFO_NOT_FOUND;
				goto RET;
			}

			// Parse dimension
			strId = cnn_strdef_get_id((const char*)xmlNodeGetContent(dim->children));
			switch(strId)
			{
				case CNN_STR_1D:
					cfgPtr->layerCfg[tmpIndex].conv.dim = CNN_DIM_1D;
					break;

				case CNN_STR_2D:
					cfgPtr->layerCfg[tmpIndex].conv.dim = CNN_DIM_2D;
					break;

				default:
					assert(!"Invalid dimension type");
			}

			// Parse size
			cnn_run(cnn_strtoi(&cfgPtr->layerCfg[tmpIndex].conv.size,
						(const char*)xmlNodeGetContent(size->children)),
					ret, RET);
			break;

		case CNN_LAYER_POOL:
			if(poolType == NULL || dim == NULL || size == NULL)
			{
				ret = CNN_INFO_NOT_FOUND;
				goto RET;
			}

			// Parse pooling type
			strId = cnn_strdef_get_id((const char*)xmlNodeGetContent(poolType->children));
			switch(strId)
			{
				case CNN_STR_MAX:
					cfgPtr->layerCfg[tmpIndex].pool.poolType = CNN_POOL_MAX;
					break;

				case CNN_STR_AVG:
					cfgPtr->layerCfg[tmpIndex].pool.poolType = CNN_POOL_AVG;
					break;

				default:
					assert(!"Invalid pooling type");
			}

			// Parse dimension
			strId = cnn_strdef_get_id((const char*)xmlNodeGetContent(dim->children));
			switch(strId)
			{
				case CNN_STR_1D:
					cfgPtr->layerCfg[tmpIndex].pool.dim = CNN_DIM_1D;
					break;

				case CNN_STR_2D:
					cfgPtr->layerCfg[tmpIndex].pool.dim = CNN_DIM_2D;
					break;

				default:
					assert(!"Invalid dimension type");
			}

			// Parse size
			cnn_run(cnn_strtoi(&cfgPtr->layerCfg[tmpIndex].pool.size,
						(const char*)xmlNodeGetContent(size->children)),
					ret, RET);

			break;

		default:
			assert(!"Invalid layer type");
	}

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

	// Parse network config
	cnn_run(cnn_parse_network_xml(cfgPtr, net), ret, RET);

RET:
	if(doc != NULL)
	{
		xmlFreeDoc(doc);
	}

	return ret;
}
