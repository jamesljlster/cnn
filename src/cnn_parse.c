#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cnn_parse.h"
#include "cnn_strdef.h"
#include "cnn_builtin_math.h"

int cnn_config_import(cnn_config_t* cfgPtr, const char* fPath)
{
	int ret = CNN_NO_ERROR;
	cnn_config_t tmpCfg = NULL;

	// Create config
	cnn_run(cnn_config_create(&tmpCfg), ret, RET);

	// Import config
	cnn_run(cnn_import_root(tmpCfg, NULL, fPath), ret, ERR);

	// Assign value
	*cfgPtr = tmpCfg;

	goto RET;

ERR:
	cnn_config_delete(tmpCfg);

RET:
	return ret;
}

int cnn_import(cnn_t* cnnPtr, const char* fPath)
{
	int ret = CNN_NO_ERROR;
	cnn_t tmpCnn = NULL;

	// Memory allocation
	cnn_alloc(tmpCnn, 1, struct CNN, ret, RET);

	// Import
	cnn_run(cnn_import_root(&tmpCnn->cfg, &tmpCnn->layerList, fPath), ret, ERR);

	// Assign value
	*cnnPtr = tmpCnn;

	goto RET;

ERR:
	cnn_delete(tmpCnn);

RET:
	return ret;
}

int cnn_parse_network_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node)
{
	int ret = CNN_NO_ERROR;
	int strId;
	int tmp;

	xmlNodePtr cur;
	xmlAttrPtr attrCur, layers;

	xmlChar* xStr = NULL;

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
	xStr = xmlNodeGetContent(layers->children);
	cnn_run(cnn_strtoi(&tmp, (const char*)xStr), ret, RET);
	xmlFree(xStr);
	xStr = NULL;

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
	xmlFree(xStr);
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

	xmlChar* xStr = NULL;

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
	xStr = xmlNodeGetContent(index->children);
	cnn_run(cnn_strtoi(&tmpIndex, (const char*)xStr), ret, RET);
	xmlFree(xStr);

	xStr = xmlNodeGetContent(type->children);
	strId = cnn_strdef_get_id((const char*)xStr);
	xmlFree(xStr);
	xStr = NULL;

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
			xStr = xmlNodeGetContent(size->children);
			cnn_run(cnn_strtoi(&cfgPtr->layerCfg[tmpIndex].fc.size, (const char*)xStr),
					ret, RET);
			xmlFree(xStr);
			xStr = NULL;
			break;

		case CNN_LAYER_AFUNC:
			if(id == NULL)
			{
				ret = CNN_INFO_NOT_FOUND;
				goto RET;
			}

			// Parse id
			xStr = xmlNodeGetContent(id->children);
			strId = cnn_get_afunc_id((const char*)xStr);
			xmlFree(xStr);
			xStr = NULL;

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
			xStr = xmlNodeGetContent(dim->children);
			strId = cnn_strdef_get_id((const char*)xStr);
			xmlFree(xStr);
			xStr = NULL;

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
			xStr = xmlNodeGetContent(size->children);
			cnn_run(cnn_strtoi(&cfgPtr->layerCfg[tmpIndex].conv.size, (const char*)xStr),
					ret, RET);
			xmlFree(xStr);
			xStr = NULL;
			break;

		case CNN_LAYER_POOL:
			if(poolType == NULL || dim == NULL || size == NULL)
			{
				ret = CNN_INFO_NOT_FOUND;
				goto RET;
			}

			// Parse pooling type
			xStr = xmlNodeGetContent(poolType->children);
			strId = cnn_strdef_get_id((const char*)xStr);
			xmlFree(xStr);
			xStr = NULL;

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
			xStr = xmlNodeGetContent(dim->children);
			strId = cnn_strdef_get_id((const char*)xStr);
			xmlFree(xStr);
			xStr = NULL;
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
			xStr = xmlNodeGetContent(size->children);
			cnn_run(cnn_strtoi(&cfgPtr->layerCfg[tmpIndex].pool.size, (const char*)xStr),
					ret, RET);
			xmlFree(xStr);
			xStr = NULL;

			break;

		default:
			assert(!"Invalid layer type");
	}

RET:
	xmlFree(xStr);
	return ret;
}

int cnn_parse_config_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node)
{
	int ret = CNN_NO_ERROR;
	int strId;

	xmlNodePtr cur;
	xmlNodePtr input; // Must exist
	xmlNodePtr lRate, batch;

	xmlChar* xStr = NULL;

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
		xStr = xmlNodeGetContent(batch);
		cnn_run(cnn_strtoi(&cfgPtr->batch, (const char*)xStr), ret, RET);
		xmlFree(xStr);
		xStr = NULL;
	}

	// Parse learning rate
	if(lRate != NULL)
	{
		xStr = xmlNodeGetContent(lRate);
		cnn_run(cnn_strtof(&cfgPtr->lRate, (const char*)xStr), ret, RET);
		xmlFree(xStr);
		xStr = NULL;
	}

RET:
	xmlFree(xStr);
	return ret;
}

int cnn_parse_config_input_xml(struct CNN_CONFIG* cfgPtr, xmlNodePtr node)
{
	int ret = CNN_NO_ERROR;
	int strId;

	xmlNodePtr cur;
	xmlNodePtr width, height, channel;

	xmlChar* xStr = NULL;

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
	xStr = xmlNodeGetContent(width);
	cnn_run(cnn_strtoi(&cfgPtr->width, (const char*)xStr), ret, RET);
	xmlFree(xStr);

	xStr = xmlNodeGetContent(height);
	cnn_run(cnn_strtoi(&cfgPtr->height, (const char*)xStr), ret, RET);
	xmlFree(xStr);

	xStr = xmlNodeGetContent(channel);
	cnn_run(cnn_strtoi(&cfgPtr->channel, (const char*)xStr), ret, RET);

RET:
	xmlFree(xStr);
	return ret;
}

int cnn_import_root(struct CNN_CONFIG* cfgPtr, union CNN_LAYER** layerPtr, const char* fPath)
{
	int ret = CNN_NO_ERROR;
	int strId;

	struct CNN tmpCnn;

	xmlDocPtr doc = NULL;
	xmlNodePtr cur, cfg, net;

	// Zero memory
	memset(&tmpCnn, 0, sizeof(struct CNN));

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

	// Parse network detail
	if(layerPtr != NULL)
	{
		// Allocate network
		cnn_run(cnn_config_struct_clone(&tmpCnn.cfg, cfgPtr), ret, RET);
		cnn_run(cnn_network_alloc(&tmpCnn), ret, RET);

		// Parsing
		cnn_run(cnn_parse_network_detail_xml(&tmpCnn, doc), ret, RET);

		// Assign value
		*layerPtr = tmpCnn.layerList;
		tmpCnn.layerList = NULL;
	}

RET:
	if(doc != NULL)
	{
		xmlFreeDoc(doc);
	}

	cnn_struct_delete(&tmpCnn);

	return ret;
}

int cnn_parse_network_detail_xml(struct CNN* cnn, xmlDocPtr doc)
{
	int i;
	int ret = CNN_NO_ERROR;

	char buf[CNN_XML_BUFLEN] = {0};

	xmlXPathContextPtr cont;
	xmlXPathObjectPtr obj = NULL;

	// Create XPath context
	cont = xmlXPathNewContext(doc);
	if(cont == NULL)
	{
		ret = CNN_MEM_FAILED;
		goto RET;
	}

	// Parse layer detail
	for(i = 0; i < cnn->cfg.layers; i++)
	{
		switch(cnn->cfg.layerCfg[i].type)
		{
			case CNN_LAYER_FC:
				// Select node
				ret = snprintf(buf, CNN_XML_BUFLEN, "/%s/%s/%s[@%s='%d']",
						cnn_str_list[CNN_STR_MODEL],
						cnn_str_list[CNN_STR_NETWORK], cnn_str_list[CNN_STR_LAYER],
						cnn_str_list[CNN_STR_INDEX], i);
				assert(ret > 0 && ret < CNN_XML_BUFLEN && "Insufficient buffer size");

				obj = xmlXPathEval((xmlChar*)buf, cont);
				if(obj == NULL)
				{
					ret = CNN_MEM_FAILED;
					goto RET;
				}

				// Parse
				if(!xmlXPathNodeSetIsEmpty(obj->nodesetval))
				{
					cnn_run(cnn_parse_network_detail_fc_xml(cnn, i,
								obj->nodesetval->nodeTab[0]),
							ret, RET);
				}

				break;

			case CNN_LAYER_CONV:
				// Select node
				ret = snprintf(buf, CNN_XML_BUFLEN, "/%s/%s/%s[@%s='%d']",
						cnn_str_list[CNN_STR_MODEL],
						cnn_str_list[CNN_STR_NETWORK], cnn_str_list[CNN_STR_LAYER],
						cnn_str_list[CNN_STR_INDEX], i);
				assert(ret > 0 && ret < CNN_XML_BUFLEN && "Insufficient buffer size");

				obj = xmlXPathEval((xmlChar*)buf, cont);
				if(obj == NULL)
				{
					ret = CNN_MEM_FAILED;
					goto RET;
				}

				// Parse
				if(!xmlXPathNodeSetIsEmpty(obj->nodesetval))
				{
					cnn_run(cnn_parse_network_detail_conv_xml(cnn, i,
								obj->nodesetval->nodeTab[0]),
							ret, RET);
				}

				break;
		}

		if(obj != NULL)
		{
			xmlXPathFreeObject(obj);
			obj = NULL;
		}
	}

RET:
	if(cont != NULL)
	{
		xmlXPathFreeContext(cont);
	}

	return ret;
}

int cnn_parse_network_detail_fc_xml(struct CNN* cnn, int layerIndex, xmlNodePtr node)
{
	int ret = CNN_NO_ERROR;
	int strId;

	xmlNodePtr cur;

	// Parsing
	cur = node->xmlChildrenNode;
	while(cur != NULL)
	{
		strId = cnn_strdef_get_id((const char*)cur->name);
		switch(strId)
		{
			case CNN_STR_WEIGHT:
				cnn_run(cnn_parse_mat(&cnn->layerList[layerIndex].fc.weight, cur),
						ret, RET);
				break;

			case CNN_STR_BIAS:
				cnn_run(cnn_parse_mat(&cnn->layerList[layerIndex].fc.bias, cur),
						ret, RET);
				break;
		}

		cur = cur->next;
	}

RET:
	return ret;
}

int cnn_parse_network_detail_conv_xml(struct CNN* cnn, int layerIndex, xmlNodePtr node)
{
	int ret = CNN_NO_ERROR;
	int strId;

	xmlNodePtr cur;

	// Parsing
	cur = node->xmlChildrenNode;
	while(cur != NULL)
	{
		strId = cnn_strdef_get_id((const char*)cur->name);
		switch(strId)
		{
			case CNN_STR_KERNEL:
				cnn_run(cnn_parse_mat(&cnn->layerList[layerIndex].conv.kernel, cur),
						ret, RET);
				break;

			case CNN_STR_BIAS:
				cnn_run(cnn_parse_mat(&cnn->layerList[layerIndex].conv.bias, cur),
						ret, RET);
				break;
		}

		cur = cur->next;
	}

RET:
	return ret;
}

int cnn_parse_mat(struct CNN_MAT* mat, xmlNodePtr node)
{
	int ret = CNN_NO_ERROR;
	int strId;
	int tmpIndex;

	xmlNodePtr cur;

	xmlChar* xStr = NULL;

	cur = node->xmlChildrenNode;
	while(cur != NULL)
	{
		strId = cnn_strdef_get_id((const char*)cur->name);
		switch(strId)
		{
			case CNN_STR_VALUE:
				// Get value index
				xStr = xmlGetProp(cur, (xmlChar*)cnn_str_list[CNN_STR_INDEX]);
				cnn_run(cnn_strtoi(&tmpIndex, (const char*)xStr), ret, RET);
				xmlFree(xStr);
				xStr = NULL;

				// Parse value
				if(tmpIndex >= 0 && tmpIndex < mat->rows * mat->cols)
				{
					xStr = xmlNodeGetContent(cur);
					cnn_run(cnn_strtof(&mat->mat[tmpIndex], (const char*)xStr), ret, RET);
					xmlFree(xStr);
					xStr = NULL;
				}

				break;
		}

		cur = cur->next;
	}

RET:
	xmlFree(xStr);
	return ret;
}
