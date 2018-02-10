
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

	// Write root
	cnn_xml_run(xmlTextWriterStartElement(xmlWriter, (xmlChar*)cnn_str_list[CNN_STR_MODEL]),
			ret, RET);

	// Write config
	cnn_run(cnn_write_config_xml(cfg, xmlWriter), ret, RET);

	// End root
	cnn_xml_run(xmlTextWriterEndElement(xmlWriter), ret, RET);

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

	// End config node
	cnn_xml_run(xmlTextWriterEndElement(writer), ret, RET);

RET:
	return ret;
}

