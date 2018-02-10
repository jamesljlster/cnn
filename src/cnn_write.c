#include <assert.h>

#include "cnn_builtin_math.h"
#include "cnn_write.h"
#include "cnn_strdef.h"

int cnn_config_export(cnn_config_t cfg, const char* fPath)
{
	return cnn_export_root(cfg, NULL, fPath);
}

int cnn_write_config_xml(struct CNN_CONFIG* cfgRef, xmlTextWriterPtr writer)
{
	int ret = CNN_NO_ERROR;
	char buf[CNN_XML_BUFLEN] = {0};

	// Start config node
	cnn_xml_run(xmlTextWriterStartElement(writer, (xmlChar*)cnn_str_list[CNN_STR_CONFIG]),
			ret, RET);

	// Start input node
	cnn_xml_run(xmlTextWriterStartElement(writer, (xmlChar*)cnn_str_list[CNN_STR_INPUT]),
			ret, RET);

	// Write width
	cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->width);
	cnn_xml_run(xmlTextWriterWriteElement(writer, (xmlChar*)cnn_str_list[CNN_STR_WIDTH],
			(xmlChar*)buf), ret, RET);

	// Write height
	cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->height);
	cnn_xml_run(xmlTextWriterWriteElement(writer, (xmlChar*)cnn_str_list[CNN_STR_HEIGHT],
			(xmlChar*)buf), ret, RET);

	// Write channel
	cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->channel);
	cnn_xml_run(xmlTextWriterWriteElement(writer, (xmlChar*)cnn_str_list[CNN_STR_CHANNEL],
			(xmlChar*)buf), ret, RET);

	// End input node
	cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);

	// Write batch
	cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->batch);
	cnn_xml_run(xmlTextWriterWriteElement(writer, (xmlChar*)cnn_str_list[CNN_STR_BATCH],
				(xmlChar*)buf), ret, RET);

	// Write learning rate
	cnn_ftostr(buf, CNN_XML_BUFLEN, cfgRef->lRate);
	cnn_xml_run(xmlTextWriterWriteElement(writer, (xmlChar*)cnn_str_list[CNN_STR_LRATE],
				(xmlChar*)buf), ret, RET);

	// End config node
	cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);

RET:
	return ret;
}

int cnn_write_layer_input_xml(xmlTextWriterPtr writer)
{
	int ret = CNN_NO_ERROR;

	// Write layer type
	cnn_xml_run(xmlTextWriterWriteAttribute(writer, (xmlChar*)cnn_str_list[CNN_STR_TYPE],
				(xmlChar*)cnn_str_list[CNN_STR_INPUT]), ret, RET);

RET:
	return ret;
}

int cnn_write_layer_fc_xml(struct CNN_CONFIG* cfgRef, union CNN_LAYER* layerRef,
		int layerIndex, xmlTextWriterPtr writer)
{
	int ret = CNN_NO_ERROR;
	char buf[CNN_XML_BUFLEN] = {0};

	// Write layer type
	cnn_xml_run(xmlTextWriterWriteAttribute(writer,
				(xmlChar*)cnn_str_list[CNN_STR_TYPE],
				(xmlChar*)cnn_str_list[CNN_STR_FC]),
			ret, RET);

	// Write size
	cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].fc.size);
	cnn_xml_run(xmlTextWriterWriteAttribute(writer,
				(xmlChar*)cnn_str_list[CNN_STR_SIZE],
				(xmlChar*)buf),
			ret, RET);

RET:
	return ret;
}

int cnn_write_layer_afunc_xml(struct CNN_CONFIG* cfgRef, int layerIndex,
		xmlTextWriterPtr writer)
{
	int ret = CNN_NO_ERROR;

	// Write layer type
	cnn_xml_run(xmlTextWriterWriteAttribute(writer, (xmlChar*)cnn_str_list[CNN_STR_TYPE],
				(xmlChar*)cnn_str_list[CNN_STR_AFUNC]),
			ret, RET);

	// Write activation function id
	cnn_xml_run(xmlTextWriterWriteAttribute(writer, (xmlChar*)cnn_str_list[CNN_STR_ID],
				(xmlChar*)cnn_afunc_name[cfgRef->layerCfg[layerIndex].aFunc.id]),
			ret, RET);

RET:
	return ret;
}

int cnn_write_layer_conv_xml(struct CNN_CONFIG* cfgRef, union CNN_LAYER* layerRef,
		int layerIndex, xmlTextWriterPtr writer)
{
	int ret = CNN_NO_ERROR;
	char buf[CNN_XML_BUFLEN] = {0};

	// Write layer type
	cnn_xml_run(xmlTextWriterWriteAttribute(writer, (xmlChar*)cnn_str_list[CNN_STR_TYPE],
				(xmlChar*)cnn_str_list[CNN_STR_CONV]),
			ret, RET);

	// Write dimension
	cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].conv.dim);
	cnn_xml_run(xmlTextWriterWriteAttribute(writer, (xmlChar*)cnn_str_list[CNN_STR_DIM],
				(xmlChar*)buf),
			ret, RET);

	// Write size
	cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].conv.size);
	cnn_xml_run(xmlTextWriterWriteAttribute(writer, (xmlChar*)cnn_str_list[CNN_STR_SIZE],
				(xmlChar*)buf),
			ret, RET);

RET:
	return ret;
}

int cnn_write_layer_pool_xml(struct CNN_CONFIG* cfgRef, int layerIndex,
		xmlTextWriterPtr writer)
{
	int ret = CNN_NO_ERROR;
	char buf[CNN_XML_BUFLEN] = {0};

	// Write layer type
	cnn_xml_run(xmlTextWriterWriteAttribute(writer, (xmlChar*)cnn_str_list[CNN_STR_TYPE],
				(xmlChar*)cnn_str_list[CNN_STR_POOL]),
			ret, RET);

	// Write pooling type
	switch(cfgRef->layerCfg[layerIndex].pool.poolType)
	{
		case CNN_POOL_MAX:
			cnn_xml_run(xmlTextWriterWriteAttribute(writer,
						(xmlChar*)cnn_str_list[CNN_STR_POOL_TYPE],
						(xmlChar*)cnn_str_list[CNN_STR_MAX]),
					ret, RET);
			break;

		case CNN_POOL_AVG:
			cnn_xml_run(xmlTextWriterWriteAttribute(writer,
						(xmlChar*)cnn_str_list[CNN_STR_POOL_TYPE],
						(xmlChar*)cnn_str_list[CNN_STR_MIN]),
					ret, RET);
			break;

		default:
			assert(!"Invalid pooling type");
	}

	// Write size
	cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[layerIndex].pool.size);
	cnn_xml_run(xmlTextWriterWriteAttribute(writer,
				(xmlChar*)cnn_str_list[CNN_STR_SIZE],
				(xmlChar*)buf),
			ret, RET);

RET:
	return ret;
}

int cnn_write_network_xml(struct CNN_CONFIG* cfgRef, union CNN_LAYER* layerRef,
		xmlTextWriterPtr writer)
{
	int ret = CNN_NO_ERROR;
	char buf[CNN_XML_BUFLEN] = {0};

	// Start network node
	cnn_xml_run(xmlTextWriterStartElement(writer, (xmlChar*)cnn_str_list[CNN_STR_NETWORK]),
				ret, RET);

	// Write size (layers)
	cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layers);
	cnn_xml_run(xmlTextWriterWriteAttribute(writer,
				(xmlChar*)cnn_str_list[CNN_STR_SIZE],
				(xmlChar*)buf),
			ret, RET);

	// Write network architecture
	for(int i = 0; i < cfgRef->layers; i++)
	{
		// Start layer node
		cnn_xml_run(xmlTextWriterStartElement(writer, (xmlChar*)cnn_str_list[CNN_STR_LAYER]),
				ret, RET);

		// Write layer index
		cnn_itostr(buf, CNN_XML_BUFLEN, i);
		cnn_xml_run(xmlTextWriterWriteAttribute(writer, (xmlChar*)cnn_str_list[CNN_STR_INDEX],
					(xmlChar*)buf),
				ret, RET);

		// Write layer attribute
		switch(cfgRef->layerCfg[i].type)
		{
			case CNN_LAYER_INPUT:
				cnn_run(cnn_write_layer_input_xml(writer), ret, RET);
				break;

			case CNN_LAYER_FC:
				cnn_run(cnn_write_layer_fc_xml(cfgRef, layerRef, i, writer), ret, RET);
				break;

			case CNN_LAYER_AFUNC:
				cnn_run(cnn_write_layer_afunc_xml(cfgRef, i, writer), ret, RET);
				break;

			case CNN_LAYER_CONV:
				cnn_run(cnn_write_layer_conv_xml(cfgRef, layerRef, i, writer), ret, RET);
				break;

			case CNN_LAYER_POOL:
				cnn_run(cnn_write_layer_pool_xml(cfgRef, i, writer), ret, RET);
				break;

			default:
				assert(!"Invalid layer type");
		}

		// End layer node
		cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);
	}

	// End network node
	cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);

RET:
	return ret;
}

int cnn_export_root(struct CNN_CONFIG* cfgRef, union CNN_LAYER* layerRef, const char* fPath)
{
	int ret = CNN_NO_ERROR;

	xmlTextWriterPtr writer = NULL;

	// Create xml writer
	writer = xmlNewTextWriterFilename(fPath, 0);
	if(writer == NULL)
	{
		ret = CNN_FILE_OP_FAILED;
		goto RET;
	}

	cnn_xml_run(xmlTextWriterSetIndent(writer, 1), ret, RET);
	cnn_xml_run(xmlTextWriterStartDocument(writer, CNN_XML_VER_STR, CNN_XML_ENC_STR, NULL),
			ret, RET);

	// Write root
	cnn_xml_run(xmlTextWriterStartElement(writer, (xmlChar*)cnn_str_list[CNN_STR_MODEL]),
			ret, RET);

	// Write config
	cnn_run(cnn_write_config_xml(cfgRef, writer), ret, RET);

	// Write netowrk
	cnn_run(cnn_write_network_xml(cfgRef, layerRef, writer), ret, RET);

	// End root
	cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);
	cnn_xml_run(xmlTextWriterEndDocument(writer), ret, RET);

RET:
	if(writer != NULL)
	{
		xmlFreeTextWriter(writer);
	}

	return ret;
}

