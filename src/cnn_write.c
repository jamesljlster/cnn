#include <assert.h>

#include "cnn_builtin_math.h"
#include "cnn_write.h"
#include "cnn_strdef.h"

int cnn_config_export(cnn_config_t cfg, const char* fPath)
{
	int ret = CNN_NO_ERROR;

	xmlTextWriterPtr xmlWriter = NULL;

	// Create xml writer
	xmlWriter = xmlNewTextWriterFilename(fPath, 0);
	if(xmlWriter == NULL)
	{
		ret = CNN_FILE_OP_FAILED;
		goto RET;
	}

	cnn_xml_run(xmlTextWriterSetIndent(xmlWriter, 1), ret, RET);
	cnn_xml_run(xmlTextWriterStartDocument(xmlWriter, CNN_XML_VER_STR, CNN_XML_ENC_STR, NULL),
			ret, RET);

	// Write root
	cnn_xml_run(xmlTextWriterStartElement(xmlWriter, (xmlChar*)cnn_str_list[CNN_STR_MODEL]),
			ret, RET);

	// Write config
	cnn_run(cnn_write_config_xml(cfg, xmlWriter), ret, RET);

	// End root
	cnn_xml_run(xmlTextWriterEndElement(xmlWriter), ret, RET);
	cnn_xml_run(xmlTextWriterEndDocument(xmlWriter), ret, RET);

RET:
	if(xmlWriter != NULL)
	{
		xmlFreeTextWriter(xmlWriter);
	}

	return ret;
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

	// Start arch node
	cnn_xml_run(xmlTextWriterStartElement(writer, (xmlChar*)cnn_str_list[CNN_STR_ARCH]),
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
		cnn_xml_run(xmlTextWriterWriteAttribute(writer,
					(xmlChar*)cnn_str_list[CNN_STR_INDEX],
					(xmlChar*)buf),
				ret, RET);

		// Write layer attribute
		switch(cfgRef->layerCfg[i].type)
		{
			case CNN_LAYER_INPUT:
				// Write layer type
				cnn_xml_run(xmlTextWriterWriteAttribute(writer,
							(xmlChar*)cnn_str_list[CNN_STR_TYPE],
							(xmlChar*)cnn_str_list[CNN_STR_INPUT]),
						ret, RET);
				break;

			case CNN_LAYER_FC:
				// Write layer type
				cnn_xml_run(xmlTextWriterWriteAttribute(writer,
							(xmlChar*)cnn_str_list[CNN_STR_TYPE],
							(xmlChar*)cnn_str_list[CNN_STR_FC]),
						ret, RET);

				// Write size
				cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[i].fc.size);
				cnn_xml_run(xmlTextWriterWriteAttribute(writer,
							(xmlChar*)cnn_str_list[CNN_STR_SIZE],
							(xmlChar*)buf),
						ret, RET);

				break;

			case CNN_LAYER_AFUNC:
				// Write layer type
				cnn_xml_run(xmlTextWriterWriteAttribute(writer,
							(xmlChar*)cnn_str_list[CNN_STR_TYPE],
							(xmlChar*)cnn_str_list[CNN_STR_AFUNC]),
						ret, RET);

				// Write activation function id
				cnn_xml_run(xmlTextWriterWriteAttribute(writer,
							(xmlChar*)cnn_str_list[CNN_STR_ID],
							(xmlChar*)cnn_afunc_name[cfgRef->layerCfg[i].aFunc.id]),
						ret, RET);

				break;

			case CNN_LAYER_CONV:
				// Write layer type
				cnn_xml_run(xmlTextWriterWriteAttribute(writer,
							(xmlChar*)cnn_str_list[CNN_STR_TYPE],
							(xmlChar*)cnn_str_list[CNN_STR_CONV]),
						ret, RET);

				// Write dimension
				cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[i].conv.dim);
				cnn_xml_run(xmlTextWriterWriteAttribute(writer,
							(xmlChar*)cnn_str_list[CNN_STR_DIM],
							(xmlChar*)buf),
						ret, RET);

				// Write size
				cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[i].conv.size);
				cnn_xml_run(xmlTextWriterWriteAttribute(writer,
							(xmlChar*)cnn_str_list[CNN_STR_SIZE],
							(xmlChar*)buf),
						ret, RET);

				break;

			case CNN_LAYER_POOL:
				// Write layer type
				cnn_xml_run(xmlTextWriterWriteAttribute(writer,
							(xmlChar*)cnn_str_list[CNN_STR_TYPE],
							(xmlChar*)cnn_str_list[CNN_STR_POOL]),
						ret, RET);

				// Write pooling type
				switch(cfgRef->layerCfg[i].pool.poolType)
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
				cnn_itostr(buf, CNN_XML_BUFLEN, cfgRef->layerCfg[i].pool.size);
				cnn_xml_run(xmlTextWriterWriteAttribute(writer,
							(xmlChar*)cnn_str_list[CNN_STR_SIZE],
							(xmlChar*)buf),
						ret, RET);

				break;

			default:
				assert(!"Invalid layer type");
		}

		// End layer node
		cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);
	}

	// End arch node
	cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);

	// End config node
	cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);

RET:
	return ret;
}

